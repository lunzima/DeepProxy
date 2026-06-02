"""Anthropic Messages API 翻译层测试。"""
from __future__ import annotations

import json

import pytest

from deep_proxy.compatibility.anthropic_translator import (
    claude_request_to_openai,
    openai_error_to_claude,
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

    def test_user_tool_result_with_text_ordering(self):
        """tool 消息必须在 user text 之前，紧跟在 assistant tool_calls 之后。"""
        body = {
            "model": "x",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "f",
                     "input": {"key": "val"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a",
                     "content": "42"},
                    {"type": "text", "text": "here is the result"},
                ]},
            ],
        }
        out = claude_request_to_openai(body)
        msgs = out["messages"]
        # 检查消息顺序：assistant (tool_calls) → tool → user
        assert msgs[0]["role"] == "assistant"
        assert msgs[0].get("tool_calls")
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "toolu_a"
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == "here is the result"

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

    async def test_streaming_error_inner_shape_normalized_to_anthropic(self):
        """流式上游错误帧（iter_litellm_chunks 产出 OpenAI 形状 {message,type,param,code}）
        翻成 Anthropic error 事件时,内层须规范化为 {type,message},不泄漏 param/code
        （与非流式 openai_error_to_claude 路径一致——端点趋同）。"""
        async def fake():
            yield {"error": {"message": "boom", "type": "rate_limit_error",
                             "param": None, "code": 429}}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        err = next(_parse_event(e)[1] for e in events
                   if _parse_event(e)[0] == "error")
        assert err["type"] == "error"
        assert err["error"] == {"type": "rate_limit_error", "message": "boom"}
        assert "param" not in err["error"]
        assert "code" not in err["error"]

    async def test_error_after_open_text_block_closes_block_first(self):
        """流中途错误（已有未关闭的 text block）：规范化——先发 content_block_stop
        配对已开的块，再发 error 事件。保证 SSE 块结构平衡（每个 start 配对 stop），
        与 OpenAI 端点 data:{error}+[DONE] 的干净终止行为趋同。"""
        async def fake():
            yield {"choices": [{"delta": {"content": "part"}, "index": 0}]}
            yield {"error": {"message": "boom", "type": "api_error"}}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        assert names.count("content_block_start") == names.count("content_block_stop")
        # content_block_stop 必须在 error 之前
        assert "error" in names
        assert names.index("content_block_stop") < names.index("error")

    async def test_error_after_open_thinking_block_closes_block_first(self):
        """reasoning 块未关时错误：同样先关 thinking 块再发 error。"""
        async def fake():
            yield {"choices": [{"delta": {"reasoning_content": "think"}, "index": 0}]}
            yield {"error": {"message": "boom", "type": "api_error"}}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        assert names.count("content_block_start") == names.count("content_block_stop")
        assert names.index("content_block_stop") < names.index("error")

    async def test_heartbeat_translated_to_ping_event(self):
        """cross_consult 心跳哨兵 → Anthropic ping 事件（保持连接温热）。"""
        async def fake():
            yield {"_dp_heartbeat": True}
            yield {"choices": [{"delta": {"content": "hi"}, "index": 0,
                                "finish_reason": "stop"}]}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        names = [_parse_event(e)[0] for e in events]
        assert "ping" in names
        # 心跳不得污染内容流：内容仍正常翻译
        assert "content_block_delta" in names

    async def test_reasoning_after_text_opens_new_thinking_block(self):
        """边缘场景：reasoning_content 在 text 已开后到达。

        DeepSeek 实际流序通常 reasoning 先于 text，但本测试钉住"如果上游
        颠倒顺序，状态机不会崩——只是给 reasoning 开了第二个 block"。
        """
        async def fake():
            yield {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
            yield {"choices": [{"delta": {"reasoning_content": "thinking"}, "index": 0}]}
            yield {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        parsed = [_parse_event(e) for e in events]
        # 必须同时含 text + thinking 块开启，且无 crash
        kinds = [(name, payload.get("content_block", {}).get("type")) for name, payload in parsed]
        assert ("content_block_start", "text") in kinds
        assert ("content_block_start", "thinking") in kinds
        # message_stop 是最后一个事件
        assert parsed[-1][0] == "message_stop"

    async def test_usage_max_across_late_chunks(self):
        """usage 多次 chunk 间累加按 max() 取值，防止迟到 chunk 倒退。"""
        async def fake():
            yield {"choices": [{"delta": {"content": "hi"}, "index": 0}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            yield {"choices": [{"delta": {"content": "!"}, "index": 0}],
                   "usage": {"prompt_tokens": 90, "completion_tokens": 30}}  # 倒退
            yield {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 60}}  # 上升

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        parsed = [_parse_event(e) for e in events]
        msg_delta = next(p for n, p in parsed if n == "message_delta")
        # max() 应取最高值：input=100, output=60
        assert msg_delta["usage"]["input_tokens"] == 100
        assert msg_delta["usage"]["output_tokens"] == 60

    async def test_tool_call_arguments_non_string_mid_stream_does_not_crash(self):
        """上游若中段发非 str 的 arguments（如 dict）应被 isinstance 守卫跳过。

        旧实现 `if fn.get("arguments"): slot["arguments"] += args` 对 truthy
        非 str（dict/list）会 TypeError 崩翻译器。falsy 值（None / 0）旧代码
        碰巧 skip 不崩，所以必须用 truthy 非 str 才能真正 exercise 守卫。
        与 utils.merge_tool_call_deltas 的 isinstance(... str) 守卫语义一致。
        """
        async def fake():
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "tc1", "type": "function",
                "function": {"name": "f", "arguments": '{"x":'},
            }]}, "index": 0}]}
            # 中段 arguments=dict（truthy 非 str） — 旧代码会 TypeError
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": {"garbage": "skipped"}},
            }]}, "index": 0}]}
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "1}"},
            }]}, "index": 0}]}
            yield {"choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}]}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        parsed = [_parse_event(e) for e in events]
        # 最终 tool_use 块的 input_json_delta 应包含完整 {"x":1}
        # （中段 dict 被守卫跳过，不污染拼接结果；旧代码会在 dict 那帧 TypeError 崩流）
        input_deltas = [p for n, p in parsed
                        if n == "content_block_delta"
                        and p.get("delta", {}).get("type") == "input_json_delta"]
        assert input_deltas, "应至少产生一个 input_json_delta"
        import json as _json
        parsed_json = _json.loads(input_deltas[0]["delta"]["partial_json"])
        assert parsed_json == {"x": 1}

    async def test_reasoning_and_text_in_same_chunk_emit_in_order(self):
        """同 chunk 含 reasoning_content + content：先 emit thinking 再 emit text，
        且 text 处理会关掉刚开的 thinking 块。"""
        async def fake():
            yield {"choices": [{"delta": {
                "reasoning_content": "思考",
                "content": "回答",
            }, "index": 0}]}
            yield {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}

        events = await _collect(openai_stream_to_claude(fake(), requested_model="x"))
        parsed = [_parse_event(e) for e in events]
        names = [n for n, _ in parsed]
        # 期望顺序：message_start, thinking start, thinking_delta, thinking stop,
        #          text start, text_delta, text stop, message_delta, message_stop
        # 关键：thinking_delta 必须在 text_delta 之前
        thinking_delta_idx = next(i for i, (n, p) in enumerate(parsed)
                                  if n == "content_block_delta"
                                  and p.get("delta", {}).get("type") == "thinking_delta")
        text_delta_idx = next(i for i, (n, p) in enumerate(parsed)
                              if n == "content_block_delta"
                              and p.get("delta", {}).get("type") == "text_delta")
        assert thinking_delta_idx < text_delta_idx
        # 期间应有 thinking 的 content_block_stop
        between = parsed[thinking_delta_idx:text_delta_idx]
        assert any(n == "content_block_stop" for n, _ in between)


class TestOpenAIErrorToClaude:
    """非流式上游错误体 OpenAI 形状 → Anthropic 形状（端点行为趋同：
    /v1/messages 上游错误须返回 Anthropic 错误体，而非泄漏 OpenAI 形状）。"""

    def test_reshapes_openai_error_envelope(self):
        detail = {"error": {"message": "rate limited", "type": "rate_limit_error",
                            "param": None, "code": 429}}
        out = openai_error_to_claude(detail)
        assert out == {"type": "error",
                       "error": {"type": "rate_limit_error", "message": "rate limited"}}

    def test_string_detail_wrapped_as_api_error(self):
        out = openai_error_to_claude("代理未就绪")
        assert out["type"] == "error"
        assert out["error"]["type"] == "api_error"
        assert out["error"]["message"] == "代理未就绪"

    def test_idempotent_on_anthropic_shape(self):
        already = {"type": "error", "error": {"type": "api_error", "message": "x"}}
        assert openai_error_to_claude(already) == already


class TestAnthropicEndpointUpstreamErrorShape:
    """端点级：/v1/messages 非流式上游错误（map_litellm_error 产出 OpenAI 形状
    HTTPException）必须以 Anthropic 错误体返回、状态码保持不变。"""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from fastapi import HTTPException
        from deep_proxy import main as main_mod
        from deep_proxy.config import DeepSeekConfig, ProxyConfig
        from deep_proxy.main import app
        from deep_proxy.router import DeepProxyRouter

        cfg = ProxyConfig(
            api_key=None,  # 关闭代理鉴权，专注错误体形状
            deepseek=DeepSeekConfig(api_key="sk-upstream", api_base="https://api.deepseek.com"),
        )
        main_mod.config = cfg
        main_mod.router = DeepProxyRouter(cfg)
        self._main_mod = main_mod
        self._HTTPException = HTTPException
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_upstream_429_returns_anthropic_error_body(self):
        async def _raise(*a, **k):
            raise self._HTTPException(
                status_code=429,
                detail={"error": {"message": "rate limited", "type": "rate_limit_error",
                                  "param": None, "code": 429}},
            )

        self._main_mod.router.chat_completions = _raise
        r = self.client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 429
        body = r.json()
        # Anthropic 形状：顶层 type=error + error.{type,message}，不含 OpenAI 的 param/code
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["message"] == "rate limited"
        assert "param" not in body["error"]
