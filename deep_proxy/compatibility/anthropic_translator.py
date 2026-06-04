"""Anthropic Messages API ↔ OpenAI Chat Completions 翻译层。

提供 in-process 翻译，不走内部 HTTP 调用：
- claude_request_to_openai: Anthropic POST /v1/messages 请求体 → OpenAI /v1/chat/completions
- openai_response_to_claude: OpenAI 非流式响应 → Anthropic Message 响应
- openai_stream_to_claude: OpenAI SSE chunks → Anthropic SSE 事件序列

支持范围：
- text / tool_use / tool_result content blocks（tool_result.is_error → 内容前缀错误标记）
- system 字段（string 或 text-block 数组）→ system message
- tools / tool_choice 双向映射
- stream（文本块；tool_use 流式作为单 block 在最后整体发出）
- stop_reason 双向映射（含 tool_use 块在场时强制 stop_reason=tool_use）
- usage 字段映射（input_tokens ↔ prompt_tokens、output_tokens ↔ completion_tokens）

**不支持/静默丢弃**：image / document / search_result content block（DeepSeek/MiMo 的
Anthropic 兼容端点不支持，丢弃避免上游 400）；DeepSeek 不支持的顶层字段（top_k、metadata 等）。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..utils import (
    format_sse_event as _sse_event, get_text_from_content, is_error_frame, is_heartbeat,
)


# ---------------------------------------------------------------------------
# Stop reason 映射
# ---------------------------------------------------------------------------

_OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _map_stop_reason(openai_finish: Optional[str]) -> str:
    return _OPENAI_TO_ANTHROPIC_STOP.get(openai_finish, "end_turn")


def openai_error_to_claude(detail: Any) -> Dict[str, Any]:
    """OpenAI 形状错误体 → Anthropic 形状（非流式端点错误趋同）。

    OpenAI:    {"error": {"message", "type", "param", "code"}}
    Anthropic: {"type": "error", "error": {"type", "message"}}

    非流式上游错误经 map_litellm_error 产出 OpenAI 形状 HTTPException.detail；
    /v1/messages 端点须改写成 Anthropic 形状再返回，避免向 Anthropic 客户端泄漏
    OpenAI 错误体（与本端点 auth/500 路径的 Anthropic 形状一致）。

    detail 已是 Anthropic 形状时**幂等**；非 dict（如纯字符串）包成 api_error。
    """
    err = detail.get("error") if isinstance(detail, dict) else None
    if isinstance(err, dict):
        return {"type": "error", "error": {
            "type": err.get("type") or "api_error",
            "message": err.get("message") or "",
        }}
    return {"type": "error", "error": {
        "type": "api_error",
        "message": detail if isinstance(detail, str) else str(detail),
    }}


# ---------------------------------------------------------------------------
# 请求翻译：Anthropic → OpenAI
# ---------------------------------------------------------------------------


def _convert_user_content_blocks(
    blocks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """user 消息的 content 数组 → (本消息 OpenAI content 数组, 需追加为 tool 消息的列表)。

    Anthropic 的 tool_result block 在 user 消息中出现，但 OpenAI 把它表示为
    独立的 role=tool 消息——所以要拆出来，由调用方插入到主消息序列。
    """
    openai_parts: List[Dict[str, Any]] = []
    tool_messages: List[Dict[str, Any]] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            openai_parts.append({"type": "text", "text": str(b.get("text", ""))})
        elif btype in ("image", "document", "search_result"):
            # DeepSeek Anthropic 兼容不支持图像/文档/搜索结果，静默丢弃避免上游 400
            continue
        elif btype == "tool_result":
            content = b.get("content")
            if isinstance(content, list):
                content = get_text_from_content(content)
            elif not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            # Anthropic tool_result.is_error 在 OpenAI tool 消息无对应字段——前缀错误标记，
            # 否则模型丢失"工具失败"信号、把失败当成功（影响重试/恢复行为）。
            if b.get("is_error"):
                content = f"[tool error] {content}"
            tool_messages.append({
                "role": "tool",
                "tool_call_id": b.get("tool_use_id", ""),
                "content": content,
            })
        # 其他未知 block 类型静默跳过
    return openai_parts, tool_messages


def _convert_assistant_content_blocks(
    blocks: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], str]:
    """assistant 消息的 content 数组 → (text 内容字符串, tool_calls 数组, reasoning_content)。

    Anthropic 历史 assistant 消息可能含 `thinking` block（extended thinking 回放），
    转换为 DeepSeek 的 `reasoning_content`，以便 router 的 ReasoningCache 多轮补齐。
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    thinking_parts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            text_parts.append(str(b.get("text", "")))
        elif btype == "thinking":
            thinking_parts.append(str(b.get("thinking", "")))
        elif btype == "redacted_thinking":
            # 不可见的加密 thinking — 用占位提示，避免空 reasoning_content
            thinking_parts.append("[redacted]")
        elif btype == "tool_use":
            tool_calls.append({
                "id": b.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": b.get("name", ""),
                    "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                },
            })
    return "\n".join(text_parts), tool_calls, "\n".join(thinking_parts)


def _convert_messages(
    anthropic_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in anthropic_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        if role == "user":
            parts, tool_msgs = _convert_user_content_blocks(content)
            # tool 消息必须紧跟在 assistant tool_calls 之后（OpenAI API 强制要求），
            # 因此先 emit tool_msgs，再 emit user text message。
            out.extend(tool_msgs)
            if parts:
                out.append({
                    "role": "user",
                    "content": "\n".join(p["text"] for p in parts),
                })
        elif role == "assistant":
            text, tool_calls, reasoning = _convert_assistant_content_blocks(content)
            msg: Dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if reasoning:
                # DeepSeek V4 多轮 thinking 上下文：assistant 历史回传 reasoning_content
                msg["reasoning_content"] = reasoning
            out.append(msg)
        else:
            # 未知 role 按 user 处理（保险起见保留 string 化的内容）
            out.append({"role": role, "content": get_text_from_content(content)})
    return out


def _convert_tools(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in anthropic_tools:
        if not isinstance(t, dict):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def _convert_tool_choice(tc: Any) -> Any:
    """Anthropic tool_choice → OpenAI tool_choice。"""
    if not isinstance(tc, dict):
        return tc
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "tool":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    if t == "none":
        return "none"
    return "auto"


def _convert_anthropic_thinking(thinking: Any) -> Optional[Dict[str, Any]]:
    """Anthropic thinking → DeepSeek V4 thinking。

    - {"type": "enabled", "budget_tokens": N} → {"type": "enabled"}
      （budget_tokens 是 Anthropic 概念，DeepSeek 通过 reasoning_effort 控制；
      router 会在缺省时填入 reasoning_effort=max。）
    - {"type": "disabled"} → 透传
    - 其他形态：尽力透传 type 字段，丢弃未识别子键
    """
    if not isinstance(thinking, dict):
        return None
    t = thinking.get("type")
    if t in ("enabled", "disabled"):
        out: Dict[str, Any] = {"type": t}
        # 允许显式透传 reasoning_effort（兼容混合客户端）
        if "reasoning_effort" in thinking:
            out["reasoning_effort"] = thinking["reasoning_effort"]
        return out
    return None


def claude_request_to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic POST /v1/messages 请求体 → OpenAI /v1/chat/completions 请求体。

    模型名透传（claude-* → deepseek-v4-flash 由 router.normalize_model_name 处理；
    隐含 thinking=enabled 由 router.default_thinking_type 处理）。
    Anthropic thinking 字段（若客户端显式提供）转 DeepSeek 格式（剥离 budget_tokens）。
    """
    out: Dict[str, Any] = {}

    if "model" in body:
        out["model"] = body["model"]

    # system 字段 → 前置 system message
    sys_value = body.get("system")
    messages: List[Dict[str, Any]] = []
    if sys_value:
        sys_text = get_text_from_content(sys_value)
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
    messages.extend(_convert_messages(body.get("messages") or []))
    out["messages"] = messages

    # 必填 / 直通参数
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p", "stream"):
        if k in body:
            out[k] = body[k]

    # stop_sequences → stop
    stops = body.get("stop_sequences")
    if stops:
        out["stop"] = stops

    # tools / tool_choice
    if body.get("tools"):
        out["tools"] = _convert_tools(body["tools"])
    if "tool_choice" in body:
        out["tool_choice"] = _convert_tool_choice(body["tool_choice"])

    # thinking 字段格式转换（剥离 budget_tokens）；缺省由 router 按模型名注入默认值。
    converted_thinking = _convert_anthropic_thinking(body.get("thinking"))
    if converted_thinking is not None:
        out["thinking"] = converted_thinking

    # output_config.effort → thinking.reasoning_effort
    # （DeepSeek 文档：output_config 部分支持，仅 effort 字段；其他子键忽略）
    output_config = body.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        thinking_obj = out.setdefault("thinking", {"type": "enabled"})
        # 仅当客户端没在 thinking 中显式给出 reasoning_effort 时填入
        thinking_obj.setdefault("reasoning_effort", output_config["effort"])

    # 启用流式时同时请求 usage 统计（与 OpenAI 一致）
    if out.get("stream"):
        out.setdefault("stream_options", {})["include_usage"] = True

    return out


# ---------------------------------------------------------------------------
# 响应翻译：OpenAI → Anthropic（非流式）
# ---------------------------------------------------------------------------


def _openai_message_to_anthropic_content(
    message: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """OpenAI assistant message → Anthropic content blocks。

    DeepSeek 的 reasoning_content（如存在）翻译为 Anthropic thinking block，
    放在 text/tool_use 之前（与 Anthropic 官方流顺序一致）。
    """
    blocks: List[Dict[str, Any]] = []
    # reasoning_content → thinking block（在 text 之前）
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})

    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        # 罕见：OpenAI assistant 返回多模态数组，提取 text 部分
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append({"type": "text", "text": str(part.get("text", ""))})

    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        try:
            input_obj = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            input_obj = {"_raw": fn.get("arguments")}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": fn.get("name", ""),
            "input": input_obj,
        })

    if not blocks:
        # Anthropic 规范要求 content 至少一个 block
        blocks.append({"type": "text", "text": ""})
    return blocks


def openai_response_to_claude(
    openai_response: Dict[str, Any],
    *,
    requested_model: str,
) -> Dict[str, Any]:
    """OpenAI 非流式响应 → Anthropic Message 响应。"""
    choices = openai_response.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    usage = openai_response.get("usage") or {}

    content_blocks = _openai_message_to_anthropic_content(msg)
    stop_reason = _map_stop_reason(choice.get("finish_reason"))
    # 发了 tool_use 块时 stop_reason 必为 tool_use——上游偶尔 tool_calls 配 finish_reason=stop，
    # Anthropic 客户端按 stop_reason 决定是否执行工具。
    if any(b.get("type") == "tool_use" for b in content_blocks):
        stop_reason = "tool_use"

    return {
        "id": openai_response.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model or openai_response.get("model", ""),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# ---------------------------------------------------------------------------
# 流式翻译：OpenAI SSE → Anthropic SSE
# ---------------------------------------------------------------------------


@dataclass
class _BlockState:
    """单个 content block 的开/关 + index 快照。

    把"is this block open + which idx is it" 一对状态聚合到一个对象，
    避免 _xxx_open 和 _xxx_idx 两条独立可变变量散布在状态机内部。
    """
    open: bool = False
    idx: int = 0


class _AnthropicStreamBuilder:
    """OpenAI chunk 流 → Anthropic SSE 事件的状态机。

    生命周期事件：
      message_start → content_block_start → (content_block_delta)*
                     → content_block_stop → message_delta → message_stop

    内部结构：
      - on_chunk/on_finish 是入口；具体逻辑分到 _handle_* / _emit_* 私有方法
      - thinking / text 块状态 → _BlockState 聚合
      - tool_calls 累加成 Anthropic 扁平形状（{id, name, arguments}）
        在 on_finish 时整块 emit
    """

    def __init__(self, requested_model: str) -> None:
        self._requested_model = requested_model
        self._msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        self._started: bool = False
        self._thinking = _BlockState()
        self._text = _BlockState()
        self._next_idx: int = 0
        self._finish_reason: Optional[str] = None
        self._usage_output: int = 0
        self._usage_input: int = 0  # message_delta 阶段从 OpenAI usage 写出
        self._tool_calls: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------- core

    def on_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """处理一个 chunk，返回 0..N 条 SSE 事件字符串。

        分发顺序：error → usage → message_start → reasoning → text → tool_calls
        → finish_reason 捕获。同包内 reasoning + text 共存时按此顺序串行 emit。
        """
        events: List[str] = []

        # 错误终止包（error 必须是 dict 且 choices 为空）
        if is_error_frame(chunk):
            events.extend(self._emit_error(chunk["error"]))
            return events

        # usage 累加（input/output 各取 max 避免迟到 chunk 倒退）
        self._absorb_usage(chunk.get("usage") or {})

        choices = chunk.get("choices") or []
        if not choices:
            return events

        ch = choices[0]
        delta = ch.get("delta") or {}

        events.extend(self._emit_message_start_if_needed())

        # reasoning_content → thinking_delta（在 text 之前；常规上游也是 reasoning 先到）
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning_delta, str) and reasoning_delta:
            events.extend(self._handle_reasoning(reasoning_delta))

        # text delta（自动关闭已开的 thinking 块）
        text_delta = delta.get("content")
        if isinstance(text_delta, str) and text_delta:
            events.extend(self._handle_text(text_delta))

        # tool_calls 累加（不立即 emit；在 on_finish 整块发出）
        self._accumulate_tool_calls(delta.get("tool_calls") or [])

        if ch.get("finish_reason"):
            self._finish_reason = ch["finish_reason"]

        return events

    def on_finish(self) -> List[str]:
        """流结束，返回所有终末事件（关闭块 + tool_use 发出 + message_delta/stop）。"""
        events: List[str] = []

        # 关闭仍开的 content blocks
        ev = self._close_block(self._thinking)
        if ev:
            events.append(ev)
        ev = self._close_block(self._text)
        if ev:
            events.append(ev)

        # 累加的 tool_use 整块 emit
        events.extend(self._emit_tool_use_blocks())

        # 空流兜底（无 chunk 抵达 → 给最小有效消息）
        if not self._started:
            events.extend(self._emit_empty_stream_fallback())

        events.extend(self._emit_message_end())
        return events

    # ----------------------------------------------------------- block helpers

    def _allocate_idx(self) -> int:
        """分配下一个 content_block index 并递增。"""
        idx = self._next_idx
        self._next_idx += 1
        return idx

    def _open_block(self, state: _BlockState, content_block: Dict[str, Any]) -> str:
        """标记 block 开启 + 分配 idx + 返回 content_block_start SSE 事件。"""
        state.idx = self._allocate_idx()
        state.open = True
        return _sse_event("content_block_start", {
            "type": "content_block_start",
            "index": state.idx,
            "content_block": content_block,
        })

    def _close_block(self, state: _BlockState) -> Optional[str]:
        """关闭 block（若已开），返回 content_block_stop SSE 事件；否则 None。"""
        if not state.open:
            return None
        ev = _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": state.idx,
        })
        state.open = False
        return ev

    # ----------------------------------------------------------- delta handlers

    def _handle_reasoning(self, reasoning_delta: str) -> List[str]:
        """thinking 块：未开则开 + 发 start，再发 thinking_delta。"""
        events: List[str] = []
        if not self._thinking.open:
            events.append(self._open_block(
                self._thinking,
                {"type": "thinking", "thinking": ""},
            ))
        events.append(_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self._thinking.idx,
            "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
        }))
        return events

    def _handle_text(self, text_delta: str) -> List[str]:
        """text 块：先关 thinking（若开）；text 未开则开；再发 text_delta。"""
        events: List[str] = []
        ev = self._close_block(self._thinking)
        if ev:
            events.append(ev)
        if not self._text.open:
            events.append(self._open_block(
                self._text,
                {"type": "text", "text": ""},
            ))
        events.append(_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self._text.idx,
            "delta": {"type": "text_delta", "text": text_delta},
        }))
        return events

    def _accumulate_tool_calls(self, tool_call_deltas: List[Any]) -> None:
        """累加 tool_call delta 到 Anthropic 扁平形状（{id, name, arguments}）。

        注：此处不复用 utils.merge_tool_call_deltas — 该 helper 输出 OpenAI 形状
        ({index, id, type, function:{name,arguments}})，本累加器直接累积成
        Anthropic tool_use 块的扁平形状，省去 on_finish 时的二次转换。
        canonical 语义保持一致：name 覆盖、arguments 拼接。
        """
        for tc in tool_call_deltas:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            slot = self._tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            # arguments 增量必须是 str 才拼接；非 str（如 None / dict）跳过——
            # 与 utils.merge_tool_call_deltas 的 isinstance 守卫保持一致，
            # 避免某些 provider 发 arguments: null 中段 crash 翻译器
            args_delta = fn.get("arguments")
            if isinstance(args_delta, str) and args_delta:
                slot["arguments"] += args_delta

    # ----------------------------------------------------------- emit helpers

    def _emit_message_start_if_needed(self) -> List[str]:
        """emit message_start（仅一次）。"""
        if self._started:
            return []
        self._started = True
        return [_sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": self._msg_id,
                "type": "message",
                "role": "assistant",
                "model": self._requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                # Anthropic SSE 规范要求 message_start 中 usage 字段必须存在；
                # 实际 input_tokens 在 message_delta 阶段从 OpenAI usage 补齐。
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })]

    def _emit_error(self, error: Dict[str, Any]) -> List[str]:
        """error 终止包：补 message_start（若需要）+ 关闭仍开的 content block + 发
        error 事件。

        规范化（与 OpenAI 端点 data:{error}+[DONE] 的干净终止趋同）：error 发生时
        前面可能有未关闭的 thinking / text 块——先发 content_block_stop 配对每个已开的
        块（保证 SSE 块结构平衡），再发 error 事件作终止信号（error 本身即终止，不再
        发 message_delta/message_stop）。
        """
        events = self._emit_message_start_if_needed()
        ev = self._close_block(self._thinking)
        if ev:
            events.append(ev)
        ev = self._close_block(self._text)
        if ev:
            events.append(ev)
        # 规范化内层错误形状：iter_litellm_chunks 的错误帧是 OpenAI 形状
        # （{message,type,param,code}），经 openai_error_to_claude 收敛为 Anthropic 的
        # {type,message}，不向 Anthropic 客户端泄漏 param/code（与非流式路径一致）。
        events.append(_sse_event("error", openai_error_to_claude({"error": error})))
        return events

    def _absorb_usage(self, usage: Dict[str, Any]) -> None:
        """累加 usage，input/output 各取 max 防止迟到 chunk 倒退。"""
        ct = usage.get("completion_tokens")
        if ct:
            self._usage_output = max(self._usage_output, int(ct))
        pt = usage.get("prompt_tokens")
        if pt:
            self._usage_input = max(self._usage_input, int(pt))

    def _emit_tool_use_blocks(self) -> List[str]:
        """流末整块发出累加的 tool_use 块（按原始 index 排序）。

        设计取舍：DeepSeek V4 的 tool_calls 增量缺乏稳定的 partial JSON 边界，
        这里在流末整块发出（content_block_start + 单个 input_json_delta + stop），
        而不是 Anthropic 官方规范的"逐 token input_json_delta 增量"。
        大多数 Anthropic SDK 客户端可正常消费；仅依赖渐进解析的客户端会感受到
        延迟（整块到达），不会感受到错误。
        """
        if not self._tool_calls:
            return []
        events = self._emit_message_start_if_needed()
        for orig_idx in sorted(self._tool_calls.keys()):
            slot = self._tool_calls[orig_idx]
            block_idx = self._allocate_idx()
            try:
                input_obj = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                input_obj = {"_raw": slot["arguments"]}
            events.append(_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": slot["id"] or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": slot["name"],
                    "input": {},
                },
            }))
            events.append(_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(input_obj, ensure_ascii=False),
                },
            }))
            events.append(_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            }))
        return events

    def _emit_empty_stream_fallback(self) -> List[str]:
        """空流：emit 一个最小有效消息（message_start + 空 text block）。"""
        events = self._emit_message_start_if_needed()
        events.append(_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))
        events.append(_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        }))
        return events

    def _emit_message_end(self) -> List[str]:
        """message_delta（含最终 usage + stop_reason）+ message_stop。"""
        return [
            _sse_event("message_delta", {
                "type": "message_delta",
                "delta": {
                    # 发过 tool_use 块时 stop_reason 必为 tool_use（上游偶尔 tool_calls 配
                    # finish_reason=stop）；否则按 finish_reason 映射。
                    "stop_reason": ("tool_use" if self._tool_calls
                                    else _map_stop_reason(self._finish_reason)),
                    "stop_sequence": None,
                },
                # input_tokens 从 OpenAI usage.prompt_tokens 转写补齐；
                # message_start 中占位 0（Anthropic SSE 规范要求该字段必现），
                # 首次有效值在此处 message_delta 阶段发出。
                "usage": {
                    "input_tokens": self._usage_input,
                    "output_tokens": self._usage_output,
                },
            }),
            _sse_event("message_stop", {"type": "message_stop"}),
        ]


async def openai_stream_to_claude(
    openai_chunks: AsyncIterator[Dict[str, Any]],
    *,
    requested_model: str,
) -> AsyncIterator[str]:
    """把 OpenAI 风格的 chunk dict 流翻译为 Anthropic SSE 事件序列。

    输入是业务层 dict 流（来自 router.iter_chat_chunks），不再含 SSE 协议字符串
    （`data:` 前缀 / `[DONE]` 前哨等已被协议层剥离）。状态机委托给 _AnthropicStreamBuilder。
    """
    builder = _AnthropicStreamBuilder(requested_model)
    async for chunk in openai_chunks:
        if not isinstance(chunk, dict):
            continue
        if is_heartbeat(chunk):
            # cross_consult 静默间隙的 keep-alive：翻成 Anthropic 原生 ping 事件，
            # 保持连接温热、防客户端 idle-read 超时（对应 OpenAI 路径的 SSE 注释帧）。
            yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"
            continue
        events = builder.on_chunk(chunk)
        for ev in events:
            yield ev
        # error 包后立即终止（error 必须是 dict 且 choices 为空）
        if is_error_frame(chunk):
            return
    for ev in builder.on_finish():
        yield ev
