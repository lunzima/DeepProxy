"""共享工具函数 —— 通用、模块独立的辅助函数集合。

所有本模块中的函数与 DeepProxy 业务逻辑解耦，可从任何模块导入。
"""

from __future__ import annotations

import asyncio
import hashlib as _hashlib
import json
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 随机区间抽样
# ---------------------------------------------------------------------------


def sample_in_range(lo: float, hi: float) -> float:
    """从 [lo, hi] 均匀抽样并 round 到 0.01。

    lo == hi 是预设里合法的"固定值"形态（如 top_p=[0.95,0.95]、penalties=[0,0]），
    不视为异常；lo > hi 才说明配置错误。
    """
    if lo > hi:
        logger.warning(
            "sample_in_range: lo=%.2f > hi=%.2f（配置非法），退化为定值 %.2f",
            lo, hi, lo,
        )
        return round(lo, 2)
    if lo == hi:
        return round(lo, 2)
    return round(random.uniform(lo, hi), 2)


# ---------------------------------------------------------------------------
# 指数退避重试
# ---------------------------------------------------------------------------


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    backoff_base: float,
    is_retryable: Callable[[Exception], bool],
    label: str = "",
) -> T:
    """通用指数退避重试。第 i 次重试等待 base*(2**i) ± 25% 抖动。

    Args:
        fn: 待重试的异步函数（零参数，闭包封装上下文）。
        max_retries: 最大重试次数（第 1 次重试发生在首次失败后）。
        backoff_base: 退避基数（秒）。
        is_retryable: 决定某异常是否应触发重试。
        label: 日志标签，用于区分调用来源。
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            if attempt >= max_retries or not is_retryable(e):
                raise
            delay = backoff_base * (2**attempt)
            delay *= 1.0 + random.uniform(-0.25, 0.25)
            logger.warning(
                "[%s] 第 %d 次重试，错误: %s，等待 %.2fs",
                label, attempt + 1, e, delay,
            )
            await asyncio.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# SSE 事件格式化
# ---------------------------------------------------------------------------


def format_sse_event(event_name: str, payload: Dict[str, Any]) -> str:
    """格式化 Anthropic / OpenAI 风格的 SSE 事件字符串。

    输出格式：event: {event_name}\\ndata: {json}\\n\\n
    """
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# tool_calls 增量累加
# ---------------------------------------------------------------------------


def merge_tool_call_deltas(
    existing: List[Dict[str, Any]],
    deltas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """OpenAI 流式 tool_calls 按 index 增量累加（canonical 实现）。

    流式场景中 tool_calls 以 delta 形式逐 chunk 发出，每个 delta 携带
    index 字段标识属于第几个 tool_call。本函数是项目内 OpenAI-format
    tool_call 累加的唯一规范实现，所有 stream-buffering 代码必须复用，
    避免 `+=` vs `=` 等语义分歧。

    Canonical 语义（不要改动）：
      - id:                覆盖（=）— OpenAI 每个 tool_call 只有一个 id；
                           首 chunk 给定，后续 chunk 不应重发。
      - type:              覆盖（=）— 通常仅 "function"。
      - function.name:     覆盖（=）— OpenAI 规范：name 仅在首 chunk 给定；
                           跨 chunk 拼接会重复字符（如 "get_weather" + "get_weather"）。
      - function.arguments: 拼接（+=）— arguments JSON 字符串按 chunk 流式发出，
                           必须按到达顺序拼接。

    返回值按 index 升序排列，确保跨调用确定性输出。
    """
    by_idx: Dict[int, Dict[str, Any]] = {
        tc.get("index", i): tc for i, tc in enumerate(existing)
    }
    for d in deltas:
        idx = d.get("index", 0)
        cur = by_idx.setdefault(
            idx,
            {"index": idx, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if d.get("id"):
            cur["id"] = d["id"]
        if d.get("type"):
            cur["type"] = d["type"]
        fn = d.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] = fn["name"]
        if isinstance(fn.get("arguments"), str):
            cur["function"]["arguments"] = (
                (cur["function"].get("arguments") or "") + fn["arguments"]
            )
    return [by_idx[k] for k in sorted(by_idx.keys())]


# ---------------------------------------------------------------------------
# URL / 路径工具
# ---------------------------------------------------------------------------


def strip_api_version(base: str) -> str:
    """去掉 URL 路径中的 `/v1` / `/beta` 后缀。"""
    base = base.rstrip("/")
    for suffix in ("/v1", "/beta"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


# ---------------------------------------------------------------------------
# 消息列表操作（system 消息查找与修改）
# ---------------------------------------------------------------------------


def find_system_message(messages: List[Dict[str, Any]]) -> tuple:
    """返回 (首条 system 的 index, content 文本, 是否可压缩)。

    - 无 system 消息 → (None, "", True)
    - content 是字符串 → (i, str, True)
    - content 是 list（多模态）或其他 → (i, "", False)
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return i, content, True
        return i, "", False
    return None, "", True


def append_to_system_message(
    messages: List[Dict[str, object]],
    text: str,
    *,
    dedup: bool = False,
) -> None:
    """把 text 追加到首条 system 消息末尾。

    行为：
    1. 已有 system 且 content 是字符串 → 末尾追加（双换行分隔）
    2. 已有 system 且 dedup=True 且尾部已含 text → 跳过不追加（幂等）
    3. 已有 system 但 content 是非字符串（多模态 list 等）→ 在其前插入新 system
    4. 无 system → 顶部插入新 system

    Args:
        messages: 消息列表（原地修改）。
        text: 要追加的文本。
        dedup: 是否检查 text 是否已出现在 content 中（幂等追加）。
    """
    if not text:
        return
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if dedup and text in content:
                return
            sep = "\n\n" if content else ""
            msg["content"] = f"{content}{sep}{text}"
        else:
            messages.insert(i, {"role": "system", "content": text})
        return
    messages.insert(0, {"role": "system", "content": text})


def prepend_to_system_message(
    messages: List[Dict[str, object]],
    text: str,
) -> None:
    """把 text 插入到首条 system 消息内容的最前面。

    行为：
    1. 已有 system 且 content 是字符串 → 最前面拼接（双换行分隔）
    2. 已有 system 但 content 是非字符串（多模态 list 等）→ 在其前插入新 system
    3. 无 system → 顶部插入新 system
    """
    if not text:
        return
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            sep = "\n\n" if content else ""
            msg["content"] = f"{text}{sep}{content}"
        else:
            messages.insert(i, {"role": "system", "content": text})
        return
    messages.insert(0, {"role": "system", "content": text})


# ---------------------------------------------------------------------------
# 内容提取
# ---------------------------------------------------------------------------


def get_text_from_content(content: Any) -> str:
    """从消息 content 字段提取纯文本字符串。

    OpenAI content 字段可以是：
    - 纯字符串 → 原样返回
    - list[dict]（多模态块数组）→ 提取 type=="text" 的块，换行拼接
    - 其他 → ""

    适用于 user / assistant / system 消息的 content 字段。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# hashlib 工具
# ---------------------------------------------------------------------------


def hash_str(text: str, *, prefix: str = "", algo: str = "sha256") -> str:
    """对字符串取哈希（默认 SHA-256 hexdigest）。"""
    h = _hashlib.new(algo)
    if prefix:
        h.update(prefix.encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def hash_payload(payload: dict, *, prefix: str = "", algo: str = "sha256") -> str:
    """对可 JSON 序列化的 payload 取哈希（默认 SHA-256）。

    JSON 序列化时按 key 排序以确保稳定。
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = _hashlib.new(algo)
    if prefix:
        h.update(prefix.encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 对话遍历 / 指纹工具
# ---------------------------------------------------------------------------
# 这些函数描述的是通用的"对话消息遍历"操作，原本散落在
# optimization/flash_upgrade.py，被 optimization / cross_consult / compatibility
# 三个独立业务层共用。集中放在 utils 避免跨层 import 造成虚假耦合。


def flatten_messages(
    messages: List[Dict[str, Any]],
    *,
    user_only: bool = False,
) -> str:
    """将消息列表拼为纯文本（用于评分 / 分析 / 指纹）。

    Args:
        user_only: 仅提取 role=="user" 的消息——避免 system 消息（如 QWEN.md
            项目文档）中出现的"架构""分布式"等词污染复杂度评分。
    """
    parts: List[str] = []
    for m in messages:
        if user_only and m.get("role") != "user":
            continue
        parts.append(get_text_from_content(m.get("content", "")))
    return "\n".join(parts)


def last_user_text(messages: List[Dict[str, Any]]) -> str:
    """提取最后一条 user 消息的纯文本内容；无 user 消息返回空串。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        return get_text_from_content(m.get("content", ""))
    return ""


def last_user_hash(messages: List[Dict[str, Any]]) -> str:
    """最后一条 user 消息的短哈希；空对话返回 "empty"。"""
    text = last_user_text(messages)
    return hash_str(text, algo="md5")[:8] if text else "empty"


def count_user_messages(messages: List[Dict[str, Any]]) -> int:
    """统计 user 消息数量。"""
    return sum(1 for m in messages if m.get("role") == "user")


def conversation_fingerprint(messages: List[Dict[str, Any]]) -> str:
    """跨轮次稳定的对话标识，不依赖客户端会话 ID。

    仅使用首条 user 内容[:300] 的 md5 —— 从对话第一轮就确定，永不变化。
    单用户场景碰撞概率可忽略；若真正发生（两对话首条完全相同），
    最坏情况是共享升格 / 重定向状态，成本可接受。

    注意：不使用 assistant 内容做 key，因为首轮升格触发时 assistant 尚不存在，
    后续 fingerprint 会改变，导致 UpgradeTracker / RedirectTracker 找不到对应 key。
    """
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user is None:
        return hash_str("empty", algo="md5")
    prefix = get_text_from_content(first_user.get("content", ""))[:300]
    return hash_str(prefix, algo="md5")


# ---------------------------------------------------------------------------
# 流式协议常量 / 帧谓词
# ---------------------------------------------------------------------------

SSE_DONE = "data: [DONE]\n\n"
"""OpenAI SSE 协议流结束标记。"""


def is_error_frame(chunk: Dict[str, Any]) -> bool:
    """OpenAI 风格纯错误终止帧判定：error 是 dict 且无 choices。

    iter_litellm_chunks 在上游错误时产出 {"error": {...}}（无 choices）。这是项目内
    "错误帧"的唯一规范判定，供所有流式消费/序列化方复用，避免裸字面量散布多处。
    """
    return isinstance(chunk.get("error"), dict) and not chunk.get("choices")


def is_heartbeat(chunk: Dict[str, Any]) -> bool:
    """心跳哨兵帧判定（`{"_dp_heartbeat": True}`）。

    client_stream 在等待间隙产出心跳，协议层据此发 SSE 注释帧 / Anthropic ping。
    判定集中于此，避免键名 `_dp_heartbeat` 在多个协议层硬编码。
    """
    return bool(chunk.get("_dp_heartbeat"))
