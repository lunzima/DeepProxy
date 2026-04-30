"""处理 DeepSeek reasoning_content（推理过程字段）。

V4 模型在 `thinking.type=enabled` 时响应可能包含 `reasoning_content` 字段。
V4 Thinking Mode 要求多轮对话（含 tool_call 场景）必须原样回传上一轮
assistant 的 `reasoning_content`，否则返回 HTTP 400。

本模块提供：
1. 响应处理：保留 `reasoning_content` 不 pop，同时添加 `reasoning` 兼容字段
2. 服务端缓存：把出方向的 reasoning_content 按 (content, tool_calls) 签名缓存，
   下一轮请求若客户端没回传，可静默从缓存补齐
3. Dummy 占位兜底：缓存也补不齐时，注入非空 dummy reasoning_content 并显式
   置 thinking.type=enabled，让本轮仍能走推理路径（不降级到 disabled —— 那会
   让本轮失去推理能力）
4. LiteLLM model_dump 兜底：从原始响应对象恢复被剥离的 reasoning_content
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ..utils import merge_tool_call_deltas
from .deepseek_fixes import is_thinking_disabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 响应处理
# ---------------------------------------------------------------------------


def process_reasoning_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """保留 `reasoning_content`，并写一份 `reasoning` 兼容字段（旧 SDK 用）。"""
    for choice in response.get("choices", []):
        slot = choice.get("delta") or choice.get("message")
        if not slot:
            continue
        rc = slot.get("reasoning_content")
        if rc is not None and "reasoning" not in slot:
            slot["reasoning"] = rc
    return response


def process_streaming_delta(delta: Dict[str, Any]) -> Dict[str, Any]:
    rc = delta.get("reasoning_content")
    if rc is not None and "reasoning" not in delta:
        delta["reasoning"] = rc
    return delta


def recover_reasoning_content(
    dumped: Dict[str, Any], original_response: Any
) -> Dict[str, Any]:
    """LiteLLM `model_dump()` 在某些版本会剥离 reasoning_content。

    若 dumped 缺字段而原始响应对象（litellm ModelResponse / Chunk）含此字段，
    从原始对象恢复回 dumped。流式 delta 同理。
    """
    try:
        raw_choices = getattr(original_response, "choices", None)
    except Exception:
        return dumped
    if not raw_choices:
        return dumped

    for i, choice in enumerate(dumped.get("choices", [])):
        if i >= len(raw_choices):
            break
        slot = choice.get("message") or choice.get("delta")
        if not isinstance(slot, dict) or slot.get("reasoning_content") is not None:
            continue
        try:
            raw_slot = (
                getattr(raw_choices[i], "message", None)
                or getattr(raw_choices[i], "delta", None)
            )
            rc = getattr(raw_slot, "reasoning_content", None)
            if rc is None:
                psf = getattr(raw_slot, "provider_specific_fields", None)
                if isinstance(psf, dict):
                    rc = psf.get("reasoning_content")
            if rc:
                slot["reasoning_content"] = rc
        except Exception:
            continue

    return dumped


# ---------------------------------------------------------------------------
# 服务端 reasoning_content 缓存（LRU）
# ---------------------------------------------------------------------------


def _normalize_tool_calls(tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
    """忽略 tool_call.id（每轮可能不同），仅保留可识别的函数+参数。"""
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    normalized = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        normalized.append({
            "type": tc.get("type", "function"),
            "name": fn.get("name"),
            "arguments": fn.get("arguments"),
        })
    return normalized


def _normalize_content(content: Any) -> Any:
    """content 可能是字符串、None 或 OpenAI 多模态数组；做稳定序列化。"""
    if content is None or isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def _serialize_prefix(prefix_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把对话前缀转成稳定结构，作为缓存键的一部分。

    剔除 reasoning_content（它本身就是我们要查的内容）以及
    tool_call.id（每轮可能变），保留 role/content/tool_calls/tool_call_id 角色绑定。
    """
    out = []
    for m in prefix_messages:
        if not isinstance(m, dict):
            continue
        out.append({
            "role": m.get("role"),
            "content": _normalize_content(m.get("content")),
            "tool_calls": _normalize_tool_calls(m.get("tool_calls")),
            "tool_call_id": m.get("tool_call_id"),
            "name": m.get("name"),
        })
    return out


def _signature(
    prefix_messages: List[Dict[str, Any]],
    content: Any,
    tool_calls: Any,
) -> str:
    """缓存键：(对话前缀, 当前 assistant content + normalized_tool_calls)。

    保证：
    - 同一对话同一轮（无论 reasoning_content 是否被客户端丢失）→ 同一 key
    - 不同对话前缀 → 不同 key
    - tool_call.id 变化不影响匹配
    """
    payload = {
        "prefix": _serialize_prefix(prefix_messages),
        "content": _normalize_content(content),
        "tool_calls": _normalize_tool_calls(tool_calls),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ReasoningCache:
    """LRU 缓存：(对话前缀, assistant 消息签名) → reasoning_content。

    项目假设单用户使用，因此缓存键只看消息内容；不做跨用户隔离。

    缓存仅来自本代理曾经发出的响应，不解析客户端的别名字段。
    """

    def __init__(self, max_size: int = 1024):
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._max = max_size

    def __len__(self) -> int:
        return len(self._cache)

    def remember(
        self,
        prefix_messages: List[Dict[str, Any]],
        content: Any,
        tool_calls: Any,
        reasoning_content: Optional[str],
    ) -> None:
        if not isinstance(reasoning_content, str) or not reasoning_content:
            return
        key = _signature(prefix_messages, content, tool_calls)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = reasoning_content
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def lookup(
        self,
        prefix_messages: List[Dict[str, Any]],
        content: Any,
        tool_calls: Any,
    ) -> Optional[str]:
        key = _signature(prefix_messages, content, tool_calls)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def remember_response(
        self,
        request_messages: List[Dict[str, Any]],
        response: Dict[str, Any],
    ) -> None:
        """从非流式响应记忆 reasoning_content。

        前缀就是该请求的 messages（模型生成本轮 assistant 时看到的完整上下文）。
        """
        for choice in response.get("choices", []):
            msg = choice.get("message")
            if not isinstance(msg, dict):
                continue
            self.remember(
                request_messages,
                msg.get("content"),
                msg.get("tool_calls"),
                msg.get("reasoning_content"),
            )

    def backfill(self, messages: List[Dict[str, Any]]) -> int:
        """对缺失 reasoning_content 的 assistant 消息原地从缓存补齐。

        每条 assistant 消息查询时使用它之前的 messages 作为前缀，
        从而精确匹配"该对话、该轮"产出的 reasoning_content。
        """
        n = 0
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            if msg.get("reasoning_content"):
                continue
            rc = self.lookup(messages[:i], msg.get("content"), msg.get("tool_calls"))
            if rc:
                msg["reasoning_content"] = rc
                n += 1
        return n


# ---------------------------------------------------------------------------
# 流式响应累加器（在流末尾把完整 message 写回缓存）
# ---------------------------------------------------------------------------


class StreamingReasoningAccumulator:
    """逐 chunk 累加 content / reasoning_content / tool_calls。

    流结束时把每个 choice 的累加结果按 request_messages 前缀写入 ReasoningCache。
    """

    def __init__(
        self,
        request_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._prefix = list(request_messages or [])
        self._slots: Dict[int, Dict[str, Any]] = {}

    def consume(self, chunk_dict: Dict[str, Any]) -> None:
        for choice in chunk_dict.get("choices", []) or []:
            idx = choice.get("index", 0)
            slot = self._slots.setdefault(idx, {"content": "", "reasoning_content": "", "tool_calls": None})
            delta = choice.get("delta") or {}
            c = delta.get("content")
            if isinstance(c, str):
                slot["content"] += c
            rc = delta.get("reasoning_content")
            if isinstance(rc, str):
                slot["reasoning_content"] += rc
            tcs = delta.get("tool_calls")
            if isinstance(tcs, list):
                if slot["tool_calls"] is None:
                    slot["tool_calls"] = []
                slot["tool_calls"] = merge_tool_call_deltas(slot["tool_calls"], tcs)

    def flush_to_cache(self, cache: ReasoningCache) -> None:
        for slot in self._slots.values():
            cache.remember(
                self._prefix,
                slot.get("content"),
                slot.get("tool_calls"),
                slot.get("reasoning_content"),
            )


# ---------------------------------------------------------------------------
# 多轮 trace 自愈：先从缓存补齐，补不齐注入 dummy 占位（保持 thinking=enabled）
# 缓存补不齐时注入的占位 reasoning_content。
# 设计要点：
# 1. 非空 —— 满足 DeepSeek 校验"必须传回 reasoning_content"
# 2. 如实说明现状 —— 避免被模型误读为真实推理而模仿其内容
# 3. 适度鼓励本轮深入推理 —— 让模型不要懒惰地跳过思考、不要盲信早期结论
# 4. 不指定推理风格 / 长度 —— 不污染本轮新生成的 reasoning_content
_DUMMY_REASONING = (
    "（上一轮对话的推理过程未被客户端保留。"
    "请对当前用户请求逐步骤仔细思考，对依赖的结论重新验证，"
    "而非假设此前推理正确。）"
)


def _inject_dummy_for_missing(messages: List[Dict[str, Any]]) -> int:
    """对仍缺 reasoning_content 的 assistant 消息原地填占位字符串。

    跳过完全空的 assistant 占位（无 content / tool_calls / function_call）。
    返回填充的条数。
    """
    n = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if msg.get("reasoning_content"):
            continue
        # 兼容别名也算
        alias = msg.get("reasoning")
        if isinstance(alias, str) and alias:
            msg["reasoning_content"] = alias
            n += 1
            continue
        if not any(msg.get(k) for k in ("content", "tool_calls", "function_call")):
            continue
        msg["reasoning_content"] = _DUMMY_REASONING
        n += 1
    return n


def ensure_reasoning_content_persistence(
    messages: List[Dict[str, Any]],
    body: Dict[str, Any],
    cache: Optional[ReasoningCache] = None,
) -> Dict[str, Any]:
    """V4 Thinking Mode 多轮 reasoning trace 自愈（静默执行）。

    顺序：
    1. 能从缓存补齐的补齐：用本代理上一轮缓存（按对话前缀）回填 reasoning_content
    2. 仍缺的注入 dummy 占位：让本轮请求依然能享受 thinking 模式
       （不再降级 thinking=disabled —— 那会让本轮失去推理能力）

    DeepSeek 在 thinking 模式下要求每一条 assistant 消息携带非空 reasoning_content，
    否则 400: "The reasoning_content in the thinking mode must be passed back to the API."

    用户显式 thinking.type=disabled 时跳过整个流程（disabled 模式不校验 reasoning_content）。
    """
    if cache is not None:
        cache.backfill(messages)

    thinking = body.get("thinking")
    explicitly_disabled = is_thinking_disabled(thinking)
    if explicitly_disabled:
        return body

    # 没显式禁用 → 显式启用 thinking 模式，确保 DeepSeek 接受 reasoning_content
    if not isinstance(thinking, dict):
        thinking = {}
        body["thinking"] = thinking
    thinking.setdefault("type", "enabled")

    # 缓存补不齐 → 注入 dummy 而非降级 thinking
    n = _inject_dummy_for_missing(messages)
    if n:
        logger.debug(
            "为 %d 条 assistant 消息注入 dummy reasoning_content（缓存未命中）", n
        )
    return body
