"""测试 reasoning_content 处理 + 服务端缓存（单用户场景）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from deep_proxy.compatibility.reasoning_handler import (
    ReasoningCache,
    StreamingReasoningAccumulator,
    ensure_reasoning_content_persistence,
    process_reasoning_response,
    process_streaming_delta,
    recover_reasoning_content,
)
from deep_proxy.optimization.flash_upgrade import conversation_fingerprint


class TestProcessReasoningResponse:
    def test_keeps_reasoning_content_and_adds_alias(self):
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thinking...",
                }
            }]
        }
        out = process_reasoning_response(resp)
        msg = out["choices"][0]["message"]
        assert msg["reasoning_content"] == "thinking..."
        assert msg["reasoning"] == "thinking..."

    def test_no_reasoning_content_no_op(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        out = process_reasoning_response(resp)
        assert "reasoning" not in out["choices"][0]["message"]


class TestStreamingDelta:
    def test_alias_added(self):
        delta = {"reasoning_content": "step 1"}
        process_streaming_delta(delta)
        assert delta["reasoning"] == "step 1"
        assert delta["reasoning_content"] == "step 1"


class TestReasoningCache:
    """LRU 缓存键 = (对话指纹, 对话前缀, content + normalized_tool_calls)。"""

    _FP_EMPTY = conversation_fingerprint([])

    def test_remember_lookup_and_miss(self):
        c = ReasoningCache()
        c.remember([], "hello", None, "RC1", fingerprint=self._FP_EMPTY)
        assert c.lookup([], "hello", None, fingerprint=self._FP_EMPTY) == "RC1"
        assert c.lookup([], "never", None, fingerprint=self._FP_EMPTY) is None

    def test_different_conversation_isolation(self):
        """不同对话指纹互不命中。"""
        c = ReasoningCache()
        prefix1 = [{"role": "user", "content": "话题甲"}]
        prefix2 = [{"role": "user", "content": "话题乙"}]
        fp1 = conversation_fingerprint(prefix1)
        fp2 = conversation_fingerprint(prefix2)
        c.remember(prefix1, "ans", None, "TRACE_FOR_TOPIC_1", fingerprint=fp1)
        assert c.lookup(prefix2, "ans", None, fingerprint=fp2) is None
        assert c.lookup(prefix1, "ans", None, fingerprint=fp1) == "TRACE_FOR_TOPIC_1"

    def test_tool_call_id_does_not_affect_signature(self):
        """tool_call.id 每轮可能不同，不应破坏匹配。"""
        c = ReasoningCache()
        tcs_a = [{"id": "abc", "type": "function",
                  "function": {"name": "search", "arguments": '{"q":"x"}'}}]
        tcs_b = [{"id": "xyz", "type": "function",
                  "function": {"name": "search", "arguments": '{"q":"x"}'}}]
        c.remember([], "looking up", tcs_a, "RC", fingerprint=self._FP_EMPTY)
        assert c.lookup([], "looking up", tcs_b, fingerprint=self._FP_EMPTY) == "RC"

    def test_different_tool_args_miss(self):
        c = ReasoningCache()
        c.remember([], "x", [{"function": {"name": "f", "arguments": "1"}}], "RC",
                    fingerprint=self._FP_EMPTY)
        assert c.lookup([], "x", [{"function": {"name": "f", "arguments": "2"}}],
                        fingerprint=self._FP_EMPTY) is None

    def test_prefix_role_distinguishes(self):
        c = ReasoningCache()
        prefix = [{"role": "user", "content": "Q"}]
        fp = conversation_fingerprint(prefix)
        c.remember(prefix, "ans", None, "RC_U", fingerprint=fp)
        assert c.lookup([{"role": "system", "content": "Q"}], "ans", None,
                        fingerprint=conversation_fingerprint([{"role": "system", "content": "Q"}])) is None

    def test_remember_response_with_request_messages(self):
        c = ReasoningCache()
        req = [{"role": "user", "content": "Q1"}]
        resp = {"choices": [{"message": {"content": "A1", "reasoning_content": "T1"}}]}
        c.remember_response(req, resp)
        next_msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        n = c.backfill(next_msgs)
        assert n == 1
        assert next_msgs[1]["reasoning_content"] == "T1"

    def test_empty_reasoning_content_not_stored(self):
        c = ReasoningCache()
        c.remember([], "x", None, "", fingerprint=self._FP_EMPTY)
        c.remember([], "y", None, None, fingerprint=self._FP_EMPTY)
        assert c.lookup([], "x", None, fingerprint=self._FP_EMPTY) is None
        assert c.lookup([], "y", None, fingerprint=self._FP_EMPTY) is None

    def test_lru_eviction(self):
        c = ReasoningCache(max_size=2)
        c.remember([], "a", None, "A", fingerprint=self._FP_EMPTY)
        c.remember([], "b", None, "B", fingerprint=self._FP_EMPTY)
        c.remember([], "c", None, "C", fingerprint=self._FP_EMPTY)
        assert c.lookup([], "a", None, fingerprint=self._FP_EMPTY) is None
        assert c.lookup([], "b", None, fingerprint=self._FP_EMPTY) == "B"
        assert c.lookup([], "c", None, fingerprint=self._FP_EMPTY) == "C"

    def test_lookup_promotes_to_recent(self):
        c = ReasoningCache(max_size=2)
        c.remember([], "a", None, "A", fingerprint=self._FP_EMPTY)
        c.remember([], "b", None, "B", fingerprint=self._FP_EMPTY)
        c.lookup([], "a", None, fingerprint=self._FP_EMPTY)
        c.remember([], "c", None, "C", fingerprint=self._FP_EMPTY)
        assert c.lookup([], "a", None, fingerprint=self._FP_EMPTY) == "A"
        assert c.lookup([], "b", None, fingerprint=self._FP_EMPTY) is None

    def test_backfill_uses_messages_prefix_per_assistant_msg(self):
        """每条 assistant 消息查询时使用它**之前**的 messages 作为 prefix。"""
        c = ReasoningCache()
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        fp = conversation_fingerprint(msgs)
        c.remember([{"role": "user", "content": "Q1"}], "A1", None, "T1", fingerprint=fp)
        c.remember([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ], "A2", None, "T2", fingerprint=fp)
        n = c.backfill(msgs)
        assert n == 2
        assert msgs[1]["reasoning_content"] == "T1"
        assert msgs[3]["reasoning_content"] == "T2"

    def test_backfill_skips_when_already_present(self):
        c = ReasoningCache()
        msgs = [{"role": "assistant", "content": "ans", "reasoning_content": "客户端有"}]
        fp = conversation_fingerprint(msgs)
        c.remember([], "ans", None, "from_cache", fingerprint=fp)
        n = c.backfill(msgs)
        assert n == 0
        assert msgs[0]["reasoning_content"] == "客户端有"


class TestEnsurePersistence:
    """整合：先 cache 按对话前缀补齐 → 仍缺则注入 dummy（不再降级 thinking）。"""

    def test_backfill_via_cache_keeps_thinking_enabled(self):
        c = ReasoningCache()
        prefix = [{"role": "user", "content": "hi"}]
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok",
             "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": ""}}]},
            {"role": "tool", "content": "r", "tool_call_id": "1"},
        ]
        fp = conversation_fingerprint(msgs)
        c.remember(prefix, "ok",
                   [{"function": {"name": "f", "arguments": ""}}], "TRACE",
                   fingerprint=fp)
        body = {"thinking": {"type": "enabled"}, "messages": msgs}
        out = ensure_reasoning_content_persistence(msgs, body, cache=c)
        assert msgs[1]["reasoning_content"] == "TRACE"
        assert out["thinking"] == {"type": "enabled"}

    def test_cache_miss_injects_dummy_keeps_thinking_enabled(self):
        msgs = [
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "r", "tool_call_id": "1"},
        ]
        body = {"thinking": {"type": "enabled"}, "messages": msgs}
        out = ensure_reasoning_content_persistence(msgs, body, cache=ReasoningCache())
        # thinking 保持 enabled；assistant 消息有非空占位以满足 DeepSeek 校验
        assert out["thinking"] == {"type": "enabled"}
        assert msgs[0]["reasoning_content"]

    def test_no_cache_injects_dummy(self):
        msgs = [{"role": "assistant", "tool_calls": [{"id": "1"}], "content": "x"}]
        body = {"thinking": {"type": "enabled"}}
        out = ensure_reasoning_content_persistence(msgs, body, cache=None)
        assert out["thinking"] == {"type": "enabled"}
        assert msgs[0]["reasoning_content"]

    def test_assistant_without_tool_calls_also_injected(self):
        """DeepSeek 对所有 assistant 消息都校验，不仅 tool_calls 那种。"""
        msgs = [
            {"role": "assistant", "content": "previous answer text"},
            {"role": "user", "content": "follow-up"},
        ]
        body = {"thinking": {"type": "enabled"}}
        out = ensure_reasoning_content_persistence(msgs, body, cache=None)
        assert out["thinking"] == {"type": "enabled"}
        assert msgs[0]["reasoning_content"]

    def test_existing_alias_promoted_to_reasoning_content(self):
        """客户端把 reasoning_content 装在别名 'reasoning' 字段里时，提升为正式字段。"""
        msgs = [{"role": "assistant", "tool_calls": [{"id": "1"}], "content": "x",
                 "reasoning": "客户端保留的兼容字段"}]
        body = {"thinking": {"type": "enabled"}}
        out = ensure_reasoning_content_persistence(msgs, body, cache=ReasoningCache())
        assert out["thinking"] == {"type": "enabled"}
        # reasoning 字段被提升为 reasoning_content
        assert msgs[0]["reasoning_content"] == "客户端保留的兼容字段"

    def test_empty_assistant_placeholder_not_filled(self):
        """完全空的 assistant 占位（无 content/tool_calls/function_call）跳过 dummy 注入。"""
        msgs = [{"role": "assistant"}]
        body = {"thinking": {"type": "enabled"}}
        out = ensure_reasoning_content_persistence(msgs, body, cache=None)
        assert out["thinking"] == {"type": "enabled"}
        assert "reasoning_content" not in msgs[0]

    def test_thinking_disabled_no_op(self):
        msgs = [{"role": "assistant", "tool_calls": [{"id": "1"}], "content": "x"}]
        body = {"thinking": {"type": "disabled"}}
        out = ensure_reasoning_content_persistence(msgs, body, cache=ReasoningCache())
        assert out["thinking"] == {"type": "disabled"}
        # disabled 模式下不注入 dummy（DeepSeek 不会校验）
        assert "reasoning_content" not in msgs[0]


class TestRecoverReasoningContent:
    """LiteLLM model_dump 剥离 reasoning_content 时从原始对象兜底。"""

    def test_recovers_from_message_attr(self):
        dumped = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        original = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(reasoning_content="recovered"))
        ])
        recover_reasoning_content(dumped, original)
        assert dumped["choices"][0]["message"]["reasoning_content"] == "recovered"

    def test_recovers_from_provider_specific_fields(self):
        dumped = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        original = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(
                reasoning_content=None,
                provider_specific_fields={"reasoning_content": "psf"},
            ))
        ])
        recover_reasoning_content(dumped, original)
        assert dumped["choices"][0]["message"]["reasoning_content"] == "psf"

    def test_no_op_if_already_present(self):
        dumped = {"choices": [{"message": {"role": "assistant", "reasoning_content": "kept"}}]}
        original = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(reasoning_content="other"))
        ])
        recover_reasoning_content(dumped, original)
        assert dumped["choices"][0]["message"]["reasoning_content"] == "kept"

    def test_handles_missing_choices_gracefully(self):
        dumped = {"id": "x"}
        recover_reasoning_content(dumped, SimpleNamespace(choices=None))
        assert dumped == {"id": "x"}


class TestStreamingAccumulator:
    """流式累加 → 按 request_messages 前缀写缓存。"""

    def test_accumulates_content_and_reasoning(self):
        prefix = [{"role": "user", "content": "Q"}]
        acc = StreamingReasoningAccumulator(request_messages=prefix)
        acc.consume({"choices": [{"index": 0, "delta": {"reasoning_content": "step1"}}]})
        acc.consume({"choices": [{"index": 0, "delta": {"reasoning_content": " step2"}}]})
        acc.consume({"choices": [{"index": 0, "delta": {"content": "Hello"}}]})
        acc.consume({"choices": [{"index": 0, "delta": {"content": " world"}}]})

        c = ReasoningCache()
        acc.flush_to_cache(c)
        fp = conversation_fingerprint(prefix)
        assert c.lookup(prefix, "Hello world", None, fingerprint=fp) == "step1 step2"
        # 空前缀不命中（不同对话）
        assert c.lookup([], "Hello world", None, fingerprint=TestReasoningCache._FP_EMPTY) is None

    def test_accumulates_tool_call_deltas(self):
        prefix = [{"role": "user", "content": "搜索 hello"}]
        acc = StreamingReasoningAccumulator(request_messages=prefix)
        acc.consume({"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "id": "abc", "type": "function",
                            "function": {"name": "search", "arguments": '{"q":'}}]}}]})
        acc.consume({"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]}}]})
        acc.consume({"choices": [{"index": 0, "delta": {"reasoning_content": "RC"}}]})

        c = ReasoningCache()
        acc.flush_to_cache(c)
        fp = conversation_fingerprint(prefix)
        tcs_other = [{"id": "different",
                      "function": {"name": "search", "arguments": '{"q":"hello"}'}}]
        assert c.lookup(prefix, "", tcs_other, fingerprint=fp) == "RC"
