"""Anthropic Messages API 翻译层测试。"""
from __future__ import annotations

import json

import pytest

from deep_proxy.compatibility.anthropic_translator import (
    claude_request_to_openai,
    openai_response_to_claude,
    openai_stream_to_claude,
)


# ---------------------------------------------------------------------------
# 请求翻译
# ---------------------------------------------------------------------------


class TestClaudeRequestToOpenAI:
    def test_basic_string_messages(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        out = claude_request_to_openai(body)
        # 模型名直接透传；claude-* → V4 + thinking 默认 enabled 由 router 完成
        assert out["model"] == "claude-3-5-sonnet-20241022"
        assert out["max_tokens"] == 100
        assert out["messages"] == [{"role": "user", "content": "Hello"}]

    def test_non_claude_model_passthrough(self):
        body = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        out = claude_request_to_openai(body)
        assert out["model"] == "deepseek-v4-pro"

    def test_anthropic_thinking_strips_budget_tokens(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
        out = claude_request_to_openai(body)
        assert out["thinking"] == {"type": "enabled"}

    def test_anthropic_thinking_disabled_passthrough(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],
            "thinking": {"type": "disabled"},
        }
        out = claude_request_to_openai(body)
        assert out["thinking"] == {"type": "disabled"}

    def test_assistant_thinking_block_to_reasoning_content(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Answer"},
                ]},
            ],
        }
        out = claude_request_to_openai(body)
        asst = out["messages"][1]
        assert asst["content"] == "Answer"
        assert asst["reasoning_content"] == "Let me think..."

    def test_system_string_prepended(self):
        body = {
            "model": "x",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        out = claude_request_to_openai(body)
        assert out["messages"][0] == {"role": "system", "content": "You are helpful"}
        assert out["messages"][1] == {"role": "user", "content": "Hi"}

    def test_system_text_block_array(self):
        body = {
            "model": "x",
            "system": [
                {"type": "text", "text": "Rule 1"},
                {"type": "text", "text": "Rule 2"},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        out = claude_request_to_openai(body)
        assert out["messages"][0]["role"] == "system"
        assert "Rule 1" in out["messages"][0]["content"]
        assert "Rule 2" in out["messages"][0]["content"]

    def test_user_text_blocks_flatten_to_string(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                ]},
            ],
        }
        out = claude_request_to_openai(body)
        assert out["messages"][0] == {"role": "user", "content": "A\nB"}

    def test_user_image_block_dropped(self):
        # DeepSeek Anthropic 兼容不支持图像/文档/搜索结果，需静默丢弃避免上游 400
        body = {
            "model": "x",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "abc",
                    }},
                    {"type": "document", "source": {"type": "base64", "data": "x"}},
                ]},
            ],
        }
        out = claude_request_to_openai(body)
        msg = out["messages"][0]
        assert msg["role"] == "user"
        # 仅保留 text
        assert msg["content"] == "Describe"

    def test_output_config_effort_to_thinking_reasoning_effort(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],
            "output_config": {"effort": "high"},
        }
        out = claude_request_to_openai(body)
        assert out["thinking"]["reasoning_effort"] == "high"
        assert out["thinking"]["type"] == "enabled"

    def test_explicit_thinking_reasoning_effort_wins_over_output_config(self):
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],
            "thinking": {"type": "enabled", "reasoning_effort": "max"},
            "output_config": {"effort": "low"},
        }
        out = claude_request_to_openai(body)
        # 显式 thinking.reasoning_effort 优先于 output_config.effort
        assert out["thinking"]["reasoning_effort"] == "max"

    def test_assistant_tool_use_to_openai_tool_calls(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "use tool"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Calling..."},
                    {"type": "tool_use", "id": "toolu_1",
                     "name": "search", "input": {"q": "x"}},
                ]},
            ],
        }
        out = claude_request_to_openai(body)
        asst = out["messages"][1]
        assert asst["role"] == "assistant"
        assert asst["content"] == "Calling..."
        assert len(asst["tool_calls"]) == 1
        tc = asst["tool_calls"][0]
        assert tc["id"] == "toolu_1"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"q": "x"}

    def test_user_tool_result_extracted_to_tool_message(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": "result data"},
                ]},
            ],
        }
        out = claude_request_to_openai(body)
        # tool_result 拆出来作为 role=tool 消息
        tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0] == {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": "result data",
        }

    def test_tools_translated(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"name": "search", "description": "Web search",
                 "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            ],
        }
        out = claude_request_to_openai(body)
        assert out["tools"][0]["type"] == "function"
        assert out["tools"][0]["function"]["name"] == "search"
        assert out["tools"][0]["function"]["parameters"] == {
            "type": "object", "properties": {"q": {"type": "string"}},
        }

    def test_tool_choice_mapping(self):
        body_any = {"model": "x", "messages": [], "tool_choice": {"type": "any"}}
        assert claude_request_to_openai(body_any)["tool_choice"] == "required"

        body_specific = {"model": "x", "messages": [],
                         "tool_choice": {"type": "tool", "name": "search"}}
        out = claude_request_to_openai(body_specific)
        assert out["tool_choice"] == {
            "type": "function", "function": {"name": "search"},
        }

    def test_stop_sequences_to_stop(self):
        body = {"model": "x", "messages": [], "stop_sequences": ["END"]}
        assert claude_request_to_openai(body)["stop"] == ["END"]

    def test_stream_adds_include_usage(self):
        body = {"model": "x", "messages": [], "stream": True}
        out = claude_request_to_openai(body)
        assert out["stream"] is True
        assert out["stream_options"] == {"include_usage": True}

    def test_unknown_fields_dropped(self):
        body = {
            "model": "x",
            "messages": [],
            "top_k": 40,
            "metadata": {"user_id": "u1"},
        }
        out = claude_request_to_openai(body)
        assert "top_k" not in out
        assert "metadata" not in out


# ---------------------------------------------------------------------------
# 响应翻译（非流式）
# ---------------------------------------------------------------------------


class TestOpenAIResponseToClaude:
    def test_basic_text_response(self):
        openai_resp = {
            "id": "chatcmpl-1",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        out = openai_response_to_claude(
            openai_resp, requested_model="claude-3-5-sonnet-20241022",
        )
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["model"] == "claude-3-5-sonnet-20241022"
        assert out["content"] == [{"type": "text", "text": "Hi there"}]
        assert out["stop_reason"] == "end_turn"
        assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}

    def test_finish_reason_length_to_max_tokens(self):
        resp = {
            "choices": [{
                "message": {"content": "..."},
                "finish_reason": "length",
            }],
            "usage": {},
        }
        out = openai_response_to_claude(resp, requested_model="x")
        assert out["stop_reason"] == "max_tokens"

    def test_tool_calls_to_tool_use_blocks(self):
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Searching",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search",
                                     "arguments": '{"q":"x"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        out = openai_response_to_claude(resp, requested_model="x")
        assert out["stop_reason"] == "tool_use"
        assert out["content"][0] == {"type": "text", "text": "Searching"}
        assert out["content"][1] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "search",
            "input": {"q": "x"},
        }

    def test_empty_content_yields_empty_text_block(self):
        resp = {"choices": [{"message": {"content": None},
                             "finish_reason": "stop"}], "usage": {}}
        out = openai_response_to_claude(resp, requested_model="x")
        assert out["content"] == [{"type": "text", "text": ""}]

    def test_reasoning_content_to_thinking_block(self):
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Final",
                    "reasoning_content": "Step by step",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        out = openai_response_to_claude(resp, requested_model="x")
        assert out["content"][0] == {"type": "thinking", "thinking": "Step by step"}
        assert out["content"][1] == {"type": "text", "text": "Final"}


# ---------------------------------------------------------------------------
# 流式翻译
# ---------------------------------------------------------------------------


async def _collect(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


def _parse_event(s: str) -> tuple[str, dict]:
    lines = s.strip().splitlines()
    event_name = lines[0].removeprefix("event: ").strip()
    data_line = next(l for l in lines if l.startswith("data:"))
    return event_name, json.loads(data_line.removeprefix("data:").strip())


class TestOpenAIStreamToClaude:
    async def test_text_only_stream_lifecycle(self):
        async def fake():
            yield {"choices": [{"delta": {"content": "Hel"}, "index": 0}]}
            yield {"choices": [{"delta": {"content": "lo"}, "index": 0}]}
            yield {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        assert names[0] == "message_start"
        assert names[1] == "content_block_start"
        assert names[2] == "content_block_delta"
        assert names[3] == "content_block_delta"
        assert names[-3] == "content_block_stop"
        assert names[-2] == "message_delta"
        assert names[-1] == "message_stop"

        # 文本拼起来等于 "Hello"
        deltas = [_parse_event(e)[1] for e in events
                  if _parse_event(e)[0] == "content_block_delta"]
        assert "".join(d["delta"]["text"] for d in deltas) == "Hello"

        # message_delta 含 stop_reason 与 output_tokens
        msg_delta = next(_parse_event(e)[1] for e in events
                         if _parse_event(e)[0] == "message_delta")
        assert msg_delta["delta"]["stop_reason"] == "end_turn"
        assert msg_delta["usage"]["output_tokens"] == 2

    async def test_empty_stream_emits_minimal_message(self):
        async def fake():
            return
            yield  # 显式异步生成器（永不产出）

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        assert "message_start" in names
        assert "content_block_start" in names
        assert "content_block_stop" in names
        assert names[-1] == "message_stop"

    async def test_tool_call_stream_accumulated_and_emitted(self):
        async def fake():
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "search", "arguments": '{"q":'},
            }]}}]}
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": '"hi"}'},
            }]}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        # 应有一个 tool_use content_block 的 start/delta/stop
        assert names.count("content_block_start") == 1
        assert names.count("content_block_stop") == 1

        start_payload = next(_parse_event(e)[1] for e in events
                             if _parse_event(e)[0] == "content_block_start")
        assert start_payload["content_block"]["type"] == "tool_use"
        assert start_payload["content_block"]["name"] == "search"

        delta_payload = next(_parse_event(e)[1] for e in events
                             if _parse_event(e)[0] == "content_block_delta")
        assert delta_payload["delta"]["type"] == "input_json_delta"
        assert json.loads(delta_payload["delta"]["partial_json"]) == {"q": "hi"}

        msg_delta = next(_parse_event(e)[1] for e in events
                         if _parse_event(e)[0] == "message_delta")
        assert msg_delta["delta"]["stop_reason"] == "tool_use"

    async def test_reasoning_stream_emits_thinking_block(self):
        async def fake():
            yield {"choices": [{"delta": {"reasoning_content": "Think "}, "index": 0}]}
            yield {"choices": [{"delta": {"reasoning_content": "more."}, "index": 0}]}
            yield {"choices": [{"delta": {"content": "Done"}, "index": 0}]}
            yield {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        parsed = [_parse_event(e) for e in events]
        # 顺序：message_start → thinking start → 2x thinking_delta →
        # thinking stop → text start → text_delta → text stop → message_delta → message_stop
        kinds = [(name, payload.get("content_block", {}).get("type")
                  or payload.get("delta", {}).get("type"))
                 for name, payload in parsed]
        assert ("content_block_start", "thinking") in kinds
        assert ("content_block_delta", "thinking_delta") in kinds
        assert ("content_block_start", "text") in kinds
        # thinking 块在 text 块之前关闭
        thinking_start_idx = next(i for i, (n, p) in enumerate(parsed)
                                  if n == "content_block_start"
                                  and p["content_block"]["type"] == "thinking")
        text_start_idx = next(i for i, (n, p) in enumerate(parsed)
                              if n == "content_block_start"
                              and p["content_block"]["type"] == "text")
        # 中间应有一个 thinking 的 content_block_stop
        between = parsed[thinking_start_idx:text_start_idx]
        assert any(n == "content_block_stop" for n, _ in between)

    async def test_error_frame_propagates_as_anthropic_error(self):
        async def fake():
            yield {"error": {"message": "boom", "type": "api_error"}}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        assert "error" in names
        err_payload = next(_parse_event(e)[1] for e in events
                           if _parse_event(e)[0] == "error")
        assert err_payload["error"]["message"] == "boom"
