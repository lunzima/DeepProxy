"""Anthropic Messages API ↔ OpenAI Chat Completions 翻译层。

提供 in-process 翻译，不走内部 HTTP 调用：
- claude_request_to_openai: Anthropic POST /v1/messages 请求体 → OpenAI /v1/chat/completions
- openai_response_to_claude: OpenAI 非流式响应 → Anthropic Message 响应
- openai_stream_to_claude: OpenAI SSE chunks → Anthropic SSE 事件序列

支持范围：
- text / image (data URL) / tool_use / tool_result content blocks
- system 字段（string 或 text-block 数组）→ system message
- tools / tool_choice 双向映射
- stream（文本块；tool_use 流式作为单 block 在最后整体发出）
- stop_reason 双向映射
- usage 字段映射（input_tokens ↔ prompt_tokens、output_tokens ↔ completion_tokens）

DeepSeek 不支持的字段（top_k、metadata 等）静默丢弃。
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..utils import format_sse_event as _sse_event, get_text_from_content


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
            # 经 _convert_user_content_blocks 过滤后 parts 仅含 text，
            # 扁平化为单 string（DeepSeek 不支持多模态）。
            if parts:
                out.append({
                    "role": "user",
                    "content": "\n".join(p["text"] for p in parts),
                })
            out.extend(tool_msgs)
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


class _AnthropicStreamBuilder:
    """OpenAI chunk 流 → Anthropic SSE 事件的状态机。

    提取 openai_stream_to_claude 的 10 个 nonlocal 可变变量为实例属性，
    将状态转换逻辑封装为 on_chunk / on_finish 方法，
    使状态机可独立实例化测试。

    生命周期事件：
      message_start → content_block_start → (content_block_delta)*
                     → content_block_stop → message_delta → message_stop
    """

    def __init__(self, requested_model: str) -> None:
        self._requested_model = requested_model
        self._msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # 状态变量（原 nonlocal）
        self._started: bool = False
        self._thinking_open: bool = False
        self._thinking_idx: int = 0
        self._text_open: bool = False
        self._text_idx: int = 0
        self._next_idx: int = 0
        self._finish_reason: Optional[str] = None
        self._usage_output: int = 0
        self._usage_input: int = 0  # 从 OpenAI usage.prompt_tokens 填入，message_delta 时写出
        self._tool_calls: Dict[int, Dict[str, Any]] = {}

    def _ensure_message_start(self) -> Optional[str]:
        """emit message_start（仅一次）。"""
        if self._started:
            return None
        self._started = True
        return _sse_event("message_start", {
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
        })

    def on_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """处理一个 chunk，返回 0..N 条 SSE 事件字符串。"""
        events: List[str] = []

        # 错误终止包（error 必须是 dict 且 choices 为空）
        if isinstance(chunk.get("error"), dict) and not chunk.get("choices"):
            msg_start = self._ensure_message_start()
            if msg_start:
                events.append(msg_start)
            events.append(_sse_event("error", {"type": "error", "error": chunk["error"]}))
            return events

        # usage 尾包 / 同包 usage（input/output 各自取最大值，避免迟到 chunk 倒退）
        u = chunk.get("usage") or {}
        if u:
            # max() 避免迟到 chunk 倒退（completion_tokens / prompt_tokens 均适用）
            ct = u.get("completion_tokens")
            if ct:
                self._usage_output = max(self._usage_output, int(ct))
            pt = u.get("prompt_tokens")
            if pt:
                self._usage_input = max(self._usage_input, int(pt))

        choices = chunk.get("choices") or []
        if not choices:
            return events

        ch = choices[0]
        delta = ch.get("delta") or {}

        msg_start = self._ensure_message_start()
        if msg_start:
            events.append(msg_start)

        # reasoning_content → thinking_delta（在 text 之前）
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning_delta, str) and reasoning_delta:
            if not self._thinking_open:
                self._thinking_idx = self._next_idx
                self._next_idx += 1
                self._thinking_open = True
                events.append(_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._thinking_idx,
                    "content_block": {"type": "thinking", "thinking": ""},
                }))
            events.append(_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self._thinking_idx,
                "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
            }))

        # text delta（自动关闭 thinking 块）
        text_delta = delta.get("content")
        if isinstance(text_delta, str) and text_delta:
            if self._thinking_open:
                events.append(_sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._thinking_idx,
                }))
                self._thinking_open = False
            if not self._text_open:
                self._text_idx = self._next_idx
                self._next_idx += 1
                self._text_open = True
                events.append(_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._text_idx,
                    "content_block": {"type": "text", "text": ""},
                }))
            events.append(_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self._text_idx,
                "delta": {"type": "text_delta", "text": text_delta},
            }))

        # tool_calls 增量累加（不立即 emit）
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            slot = self._tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

        if ch.get("finish_reason"):
            self._finish_reason = ch["finish_reason"]

        return events

    def on_finish(self) -> List[str]:
        """流结束，返回所有终末事件（关闭块 + tool_use 发出 + message_delta/stop）。"""
        events: List[str] = []

        # 关闭未关闭的 thinking 块
        if self._thinking_open:
            events.append(_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": self._thinking_idx,
            }))
            self._thinking_open = False

        # 关闭文本块
        if self._text_open:
            events.append(_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": self._text_idx,
            }))

        # 发出累加的 tool_use 块（按原始 index 排序）
        # 设计取舍：DeepSeek V4 的 tool_calls 增量缺乏稳定的 partial JSON 边界，
        # 这里在流末整块发出（content_block_start + 单个 input_json_delta + stop），
        # 而不是 Anthropic 官方规范的"逐 token input_json_delta 增量"。
        # 大多数 Anthropic SDK 客户端可正常消费此整体 block 形态；
        # 仅依赖 streaming tool input 渐进解析的客户端会感受到延迟（整块到达），
        # 不会感受到错误。
        if self._tool_calls:
            msg_start = self._ensure_message_start()
            if msg_start:
                events.append(msg_start)
            for orig_idx in sorted(self._tool_calls.keys()):
                slot = self._tool_calls[orig_idx]
                block_idx = self._next_idx
                self._next_idx += 1
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

        # 空流兜底：emit 一个最小有效消息
        if not self._started:
            events.append(self._ensure_message_start() or "")
            events.append(_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }))
            events.append(_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }))

        events.append(_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": _map_stop_reason(self._finish_reason),
                "stop_sequence": None,
            },
            # input_tokens 从 OpenAI usage.prompt_tokens 转写补齐；
            # message_start 中占位 0（Anthropic SSE 规范要求该字段必现），
            # 首次有效值在此处 message_delta 阶段发出。
            "usage": {
                "input_tokens": self._usage_input,
                "output_tokens": self._usage_output,
            },
        }))
        events.append(_sse_event("message_stop", {"type": "message_stop"}))

        return events


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
        events = builder.on_chunk(chunk)
        for ev in events:
            yield ev
        # error 包后立即终止（error 必须是 dict 且 choices 为空）
        if isinstance(chunk.get("error"), dict) and not chunk.get("choices"):
            return
    for ev in builder.on_finish():
        yield ev
