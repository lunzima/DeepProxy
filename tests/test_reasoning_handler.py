"""测试 reasoning_content 处理 + 服务端缓存（单用户场景）。"""
from __future__ import annotations

from types import SimpleNamespace

from deep_proxy.compatibility.reasoning_handler import (
    ReasoningCache,
    StreamingReasoningAccumulator,
    ensure_reasoning_content_persistence,
    process_reasoning_response,
    process_streaming_delta,
    recover_reasoning_content,
)
from deep_proxy.optimization.flash_upgrade import conversation_fingerprint


def test_accumulator_snapshot_restore_isolates_failed_attempt():
    acc = StreamingReasoningAccumulator(request_messages=[{"role": "user", "content": "hi"}])
    acc.consume({"choices": [{"index": 0, "delta": {"content": "turn1"}}]})
    snap = acc.snapshot()
    # 失败尝试追加内容后回滚
    acc.consume({"choices": [{"index": 0, "delta": {"reasoning_content": "junk"}}]})
    acc.restore(snap)
    assert acc._slots[0]["content"] == "turn1"
    assert acc._slots[0]["reasoning_content"] == ""
    # snapshot 深拷贝：restore 后继续 consume 不污染已持有的 snap
    acc.consume({"choices": [{"index": 0, "delta": {"content": "2"}}]})
    assert snap[0]["content"] == "turn1"


def test_accumulator_snapshot_restore_isolates_tool_calls():
    """tool_calls 嵌套结构也须隔离：merge_tool_call_deltas 原地改 inner dict，浅拷贝
    会让废弃尝试的 arguments 污染 snapshot，restore 后泄漏。"""
    acc = StreamingReasoningAccumulator(request_messages=[])
    acc.consume({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "t1", "type": "function",
         "function": {"name": "f", "arguments": '{"a":'}}]}}]})
    snap = acc.snapshot()
    # 失败尝试继续累加 tool_call arguments（merge 原地拼接 inner dict）
    acc.consume({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": '1}'}}]}}]})
    acc.restore(snap)
    args = acc._slots[0]["tool_calls"][0]["function"]["arguments"]
    assert args == '{"a":'        # 不含废弃尝试的 '1}'


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

    def test_backfill_hits_despite_rotating_user_marker(self):
        """回归：cot/inner_os marker 注入到逐轮轮转的"末条 user"不得破坏缓存键。

        生成轮该 user 是末条→带 marker；回填轮它已非末条→无 marker。键须剥离 marker、
        使 prefix 等价客户端原文，否则最近几条 assistant 永远 miss → dummy 兜底
        （生产 reasoning safety-net 补齐 0/N 即此连锁）。"""
        from deep_proxy.optimization.tool_call_chinese_cot import (
            TOOL_CALL_CN_COT_USER_MARKER as MK,
        )
        c = ReasoningCache()
        sys = {"role": "system", "content": "S"}
        u1 = {"role": "user", "content": "Q1"}
        a1 = {"role": "assistant", "content": "A1", "reasoning_content": "r1"}
        t1 = {"role": "tool", "tool_call_id": "1", "content": "tool out"}
        u2 = {"role": "user", "content": "Q2"}
        # 生成轮：u2 末条 → 带 marker；存 A2 的 reasoning
        flush_prefix = [sys, u1, a1, t1, {**u2, "content": u2["content"] + MK}]
        c.remember(flush_prefix, "A2", None, "REAL_A2",
                   fingerprint=conversation_fingerprint(flush_prefix))
        # 回填轮：u2 已非末条 → 无 marker
        backfill_prefix = [sys, u1, a1, t1, u2]
        full = [*backfill_prefix, {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"}]
        assert c.lookup(backfill_prefix, "A2", None,
                        fingerprint=conversation_fingerprint(full)) == "REAL_A2"

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

    def test_empty_assistant_placeholder_also_filled(self):
        """空 assistant（无 content/tool_calls/function_call）同样必须补 reasoning_content。

        回归：Anthropic 端 thinking-only 轮经翻译成 {role:assistant, content:None}（thinking
        被客户端剥离时无 reasoning），DeepSeek thinking 模式仍校验 reasoning_content，
        漏补即 400 'must be passed back'（生产日志 reasoning safety-net 补齐 0/3 即此）。"""
        msgs = [{"role": "assistant"}]
        body = {"thinking": {"type": "enabled"}}
        out = ensure_reasoning_content_persistence(msgs, body, cache=None)
        assert out["thinking"] == {"type": "enabled"}
        assert msgs[0]["reasoning_content"]

    def test_content_none_assistant_filled(self):
        """content=None 且无 tool_calls 的历史 assistant（翻译产物）必须补 reasoning_content。"""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "again"},
        ]
        body = {"thinking": {"type": "enabled"}}
        ensure_reasoning_content_persistence(msgs, body, cache=None)
        assert msgs[1]["reasoning_content"]

    def test_thinking_disabled_still_injects(self):
        """thinking=disabled 也必须补 reasoning_content。

        回归：LiteLLM deepseek transform 丢弃 {type:disabled}（只透传 enabled），上游
        收到"无 thinking"→ DeepSeek 默认 enabled → 仍校验 reasoning_content。早期"disabled
        即跳过"导致漏补 → 400（生产 reasoning safety-net 补齐 0/1 即此）。
        thinking 字段保留用户原值（disabled 由 LiteLLM 丢弃，不在此强改）。"""
        msgs = [{"role": "assistant", "tool_calls": [{"id": "1"}], "content": "x"}]
        body = {"thinking": {"type": "disabled"}}
        ensure_reasoning_content_persistence(msgs, body, cache=ReasoningCache())
        assert msgs[0]["reasoning_content"]

    def test_thinking_disabled_runs_cache_backfill(self):
        """disabled 时仍跑 cache.backfill——上游实际按 enabled 校验，回填真实 reasoning
        正是所需（早期"disabled 跳过 backfill"会让历史漏带 reasoning_content）。"""
        called = {"backfill": False}

        class _SpyCache:
            def backfill(self, messages):
                called["backfill"] = True

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "next"},
        ]
        body = {"thinking": {"type": "disabled"}}
        ensure_reasoning_content_persistence(msgs, body, cache=_SpyCache())
        assert called["backfill"] is True
        assert msgs[1]["reasoning_content"]


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
