"""验证 OpenAI 端点流式路径的 [DONE] 前哨、错误帧、业务层 dict 流。"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, List

import pytest


class _FakeChunk(SimpleNamespace):
    def model_dump(self) -> dict:
        return self.__dict__.copy()


async def _consume(agen: AsyncIterator) -> List:
    out = []
    async for x in agen:
        out.append(x)
    return out


class TestStreamingProtocolLayer:
    """协议层 chat_completions_stream：把业务 dict 流序列化为 SSE 字符串 + [DONE]。"""

    async def test_normal_stream_appends_done(self, router, monkeypatch):
        async def fake_acompletion(**kwargs):
            async def _gen():
                yield _FakeChunk(choices=[{"delta": {"content": "Hi"}, "index": 0}])
                yield _FakeChunk(choices=[
                    {"delta": {}, "index": 0, "finish_reason": "stop"}
                ])
            return _gen()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        chunks = await _consume(router.chat_completions_stream({"model": "deepseek-v4-flash"}))
        assert chunks[-1] == "data: [DONE]\n\n"
        assert chunks[-2].startswith("data: {")

    async def test_open_failure_emits_error_and_done(self, router, monkeypatch):
        async def fake_acompletion(**kwargs):
            raise RuntimeError("upstream 503")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        router.config.deepseek.max_retries = 0

        chunks = await _consume(router.chat_completions_stream({"model": "deepseek-v4-flash"}))
        assert chunks[-1] == "data: [DONE]\n\n"
        err_payload = chunks[-2].removeprefix("data: ").rstrip()
        assert "error" in json.loads(err_payload)

    async def test_midstream_failure_emits_error_and_done(self, router, monkeypatch):
        async def fake_acompletion(**kwargs):
            async def _gen():
                yield _FakeChunk(choices=[{"delta": {"content": "ok"}, "index": 0}])
                raise RuntimeError("connection reset")
            return _gen()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        router.config.deepseek.max_retries = 0

        chunks = await _consume(router.chat_completions_stream({"model": "deepseek-v4-flash"}))
        assert chunks[-1] == "data: [DONE]\n\n"
        err_payload = chunks[-2].removeprefix("data: ").rstrip()
        assert "error" in json.loads(err_payload)


class TestBusinessLayerDictStream:
    """业务层 iter_chat_chunks：纯 dict 流，不含协议字符串。"""

    async def test_normal_yields_dicts_no_done(self, router, monkeypatch):
        async def fake_acompletion(**kwargs):
            async def _gen():
                yield _FakeChunk(choices=[{"delta": {"content": "x"}, "index": 0}])
                yield _FakeChunk(choices=[{"delta": {}, "index": 0, "finish_reason": "stop"}])
            return _gen()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        items = await _consume(router.iter_chat_chunks({"model": "deepseek-v4-flash"}))
        # 全部是 dict，没有 [DONE] 字符串
        assert all(isinstance(i, dict) for i in items)
        assert any("choices" in i for i in items)

    async def test_error_yields_error_dict(self, router, monkeypatch):
        async def fake_acompletion(**kwargs):
            raise RuntimeError("boom")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        router.config.deepseek.max_retries = 0

        items = await _consume(router.iter_chat_chunks({"model": "deepseek-v4-flash"}))
        assert len(items) == 1
        assert "error" in items[0]


def _stream_text(items: List) -> str:
    """拼接业务 dict 流中所有 content delta。"""
    return "".join(
        c.get("delta", {}).get("content", "")
        for it in items if isinstance(it, dict)
        for c in it.get("choices", [])
    )


class TestStreamingAccumulatorSlots:
    """StreamingReasoningAccumulator.get_slot / update_slot_content（commit 引用但此前缺失）。"""

    def test_get_slot_default_when_empty(self):
        from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator
        acc = StreamingReasoningAccumulator()
        slot = acc.get_slot(0)
        assert slot == {"content": "", "reasoning_content": "", "tool_calls": None}

    def test_update_slot_content_overwrites(self):
        from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator
        acc = StreamingReasoningAccumulator()
        acc.consume({"choices": [{"index": 0, "delta": {"content": "原始违规文本"}}]})
        acc.update_slot_content(0, "修正文本", "推理")
        slot = acc.get_slot(0)
        assert slot["content"] == "修正文本"
        assert slot["reasoning_content"] == "推理"

    def test_update_slot_preserves_tool_calls_by_default(self):
        from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator
        acc = StreamingReasoningAccumulator()
        acc.consume({"choices": [{"index": 0, "delta": {
            "content": "x",
            "tool_calls": [{"index": 0, "id": "t1", "type": "function",
                            "function": {"name": "Edit", "arguments": "{}"}}],
        }}]})
        acc.update_slot_content(0, "修正", "")
        assert acc.get_slot(0)["tool_calls"][0]["id"] == "t1"  # 缺省不清空


class TestStreamingStyleGuard:
    """流式路径 StyleGuard post-stream 扫描 + 修正 + 帧重建。"""

    async def test_stream_violation_corrected(self, router, monkeypatch):
        router.config.style_guard.enabled = True

        async def fake_acompletion(**kwargs):
            async def _gen():
                yield _FakeChunk(choices=[{"delta": {"content": "他站在那里，没有动。"}, "index": 0}])
                yield _FakeChunk(choices=[{"delta": {}, "index": 0, "finish_reason": "stop"}])
            return _gen()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        async def fake_resend(config, body, *, provider=None):
            return {"choices": [{"message": {"role": "assistant",
                                             "content": "他坐在那里，双手搁在桌面。"},
                                 "finish_reason": "stop"}]}
        monkeypatch.setattr("deep_proxy.router.call_litellm", fake_resend)

        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "写一段叙事"}]}
        items = await _consume(router.iter_chat_chunks(body))
        text = _stream_text(items)
        assert "双手搁在桌面" in text
        assert "没有动" not in text  # 违规原文不应外泄给客户端

    async def test_stream_clean_passes_through(self, router, monkeypatch):
        router.config.style_guard.enabled = True

        async def fake_acompletion(**kwargs):
            async def _gen():
                yield _FakeChunk(choices=[{"delta": {"content": "他点了点头。"}, "index": 0}])
                yield _FakeChunk(choices=[{"delta": {}, "index": 0, "finish_reason": "stop"}])
            return _gen()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        async def fake_resend(config, body, *, provider=None):
            raise AssertionError("无违规不应触发重发")
        monkeypatch.setattr("deep_proxy.router.call_litellm", fake_resend)

        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "写"}]}
        items = await _consume(router.iter_chat_chunks(body))
        assert "他点了点头" in _stream_text(items)

    async def test_heartbeat_emitted_during_slow_scan(self, router, monkeypatch):
        """扫描阶段的阻塞重发期间须发心跳哨兵，防客户端 idle 超时。"""
        import asyncio
        router.config.style_guard.enabled = True
        router.config.streaming.heartbeat_seconds = 1  # 加速心跳

        async def fake_acompletion(**kwargs):
            async def _gen():
                yield _FakeChunk(choices=[{"delta": {"content": "他站在那里，没有动。"}, "index": 0}])
                yield _FakeChunk(choices=[{"delta": {}, "index": 0, "finish_reason": "stop"}])
            return _gen()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        async def slow_resend(config, body, *, provider=None):
            await asyncio.sleep(2.2)  # 跨过两个心跳间隔
            return {"choices": [{"message": {"role": "assistant",
                                             "content": "他坐在那里，翻开卷宗。"},
                                 "finish_reason": "stop"}]}
        monkeypatch.setattr("deep_proxy.router.call_litellm", slow_resend)

        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "写"}]}
        from deep_proxy.utils import is_heartbeat
        items = await _consume(router.iter_chat_chunks(body))
        assert any(isinstance(i, dict) and is_heartbeat(i) for i in items), \
            "扫描阻塞期间应至少发出一个心跳哨兵"
        assert "翻开卷宗" in _stream_text(items)

    async def test_scan_stream_fluency_reviews_tool_written_content(self, router, monkeypatch):
        """流式：tool_call 写入的叙事正文无 regex 违规时，仍走 fluency 审查并就地回写，
        且保留 tool_calls（agent 写小说到文件的核心场景）。"""
        import json as _json
        router.config.style_guard.enabled = True
        from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator

        narrative = "他走到门前，转身看向窗外。她坐下，茶杯放在桌上。夜色里，码头很暗。"
        acc = StreamingReasoningAccumulator(request_messages=[{"role": "user", "content": "写"}])
        acc.consume({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "t1", "type": "function",
            "function": {"name": "Write",
                         "arguments": _json.dumps({"file_path": "ch1.md", "content": narrative})},
        }]}}]})

        async def fake_resend(config, body, *, provider=None):
            return {"choices": [{"message": {"role": "assistant",
                                             "content": narrative + "（已润色）"}}]}
        monkeypatch.setattr("deep_proxy.router.call_litellm", fake_resend)

        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "写"}]}
        frames = await router._styleguard_scan_stream(body, None, acc, [], "tool_calls")
        # tool_calls 帧仍在，且写入正文被润色
        tc_frames = [f for f in frames
                     for c in f.get("choices", []) if c.get("delta", {}).get("tool_calls")]
        assert tc_frames, "tool_calls 帧应保留"
        args = _json.loads(tc_frames[0]["choices"][0]["delta"]["tool_calls"][0]
                           ["function"]["arguments"])
        assert args["content"] == narrative + "（已润色）"

    async def test_scan_stream_pure_toolcall_arg_violation(self, router, monkeypatch):
        """纯 tool_call（无文本）+ Edit 参数违规：流式也须扫描并修正（修复 if _content 门控）。"""
        import json as _json
        router.config.style_guard.enabled = True
        from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator

        acc = StreamingReasoningAccumulator(request_messages=[{"role": "user", "content": "写"}])
        acc.consume({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "tc1", "type": "function",
            "function": {"name": "Edit",
                         "arguments": _json.dumps({"new_string": "他站着，没有动。"})},
        }]}}]})

        async def fake_resend(config, body, *, provider=None):
            return {"choices": [{"message": {"role": "assistant",
                                             "content": "他站着，双手垂在身侧。"},
                                 "finish_reason": "stop"}]}
        monkeypatch.setattr("deep_proxy.router.call_litellm", fake_resend)

        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "写"}]}
        frames = await router._styleguard_scan_stream(body, None, acc, [], "tool_calls")
        text = "".join(
            c.get("delta", {}).get("content", "")
            for f in frames for c in f.get("choices", [])
        )
        assert "双手垂在身侧" in text  # 参数违规被修正并重新下发
        # 修正把 tool_call 改写成纯文本 → finish_reason 必须降为 stop，不能仍是 tool_calls
        finishes = [c.get("finish_reason") for f in frames for c in f.get("choices", [])]
        assert "tool_calls" not in finishes
        assert finishes[-1] == "stop"


class TestApplyStyleGuardOverride:
    """非流式 _apply_style_guard 的 override 处理须与流式一致：跳过 StyleGuard 与 fluency。"""

    async def test_override_tag_skips_styleguard_and_fluency(self, router, monkeypatch):
        router.config.style_guard.enabled = True
        # 含 override 标签 + 充足叙事锚点（无标签时会触发 fluency 重发）
        content = ("[style-override]他走到门前，转身看向窗外。她坐下，茶杯放在桌上。"
                   "夜色里，码头的灯光很暗。")
        result = {"choices": [{"message": {"role": "assistant", "content": content},
                               "finish_reason": "stop"}]}
        called = False

        async def fake_call_litellm(*a, **k):
            nonlocal called
            called = True
            return {"choices": [{"message": {"content": "不应被调用"}}]}
        monkeypatch.setattr("deep_proxy.router.call_litellm", fake_call_litellm)

        out = await router._apply_style_guard({"messages": []}, None, result)
        assert called is False  # override → 既不 StyleGuard 重发也不 fluency
        assert "[style-override]" not in out["choices"][0]["message"]["content"]


class TestMessageToChunkFrames:
    """_message_to_chunk_frames / _rebuild_stream_frames（修正消息 → 流式帧）。"""

    def test_finish_reason_derived_from_message_text(self):
        from deep_proxy.router import _message_to_chunk_frames
        frames = _message_to_chunk_frames({"role": "assistant", "content": "纯文本"})
        assert frames[-1]["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_tool_calls_when_present(self):
        from deep_proxy.router import _message_to_chunk_frames
        frames = _message_to_chunk_frames({
            "role": "assistant", "content": "",
            "tool_calls": [{"index": 0, "id": "t1", "type": "function",
                            "function": {"name": "Edit", "arguments": "{}"}}],
        })
        assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert frames[-1]["choices"][0]["delta"].get("tool_calls")

    def test_frames_share_id_and_carry_metadata(self):
        from deep_proxy.router import _message_to_chunk_frames
        frames = _message_to_chunk_frames(
            {"role": "assistant", "content": "正文",
             "tool_calls": [{"index": 0, "id": "t", "type": "function",
                             "function": {"name": "Edit", "arguments": "{}"}}]},
            frame_id="cid", model="m1", created=999,
        )
        assert {f["id"] for f in frames} == {"cid"}  # 同一 completion 共享 id
        assert all(f["model"] == "m1" and f["created"] == 999 for f in frames)
        assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
        assert "role" not in frames[1]["choices"][0]["delta"]  # role 仅首帧

    def test_rebuild_reuses_upstream_id_and_model(self):
        from deep_proxy.router import DeepProxyRouter
        buffered = [
            {"id": "up-1", "model": "deepseek-v4-flash", "created": 7,
             "choices": [{"delta": {"content": "x"}, "index": 0}]},
            {"id": "up-1", "model": "deepseek-v4-flash", "created": 7,
             "choices": [], "usage": {"total_tokens": 5}},
        ]
        out = DeepProxyRouter._rebuild_stream_frames(
            {"role": "assistant", "content": "修正"}, buffered,
        )
        assert {f["id"] for f in out} == {"up-1"}  # 沿用上游 id
        assert all(f.get("model") == "deepseek-v4-flash" for f in out)

    def test_rebuild_forwards_clean_usage_only(self):
        from deep_proxy.router import DeepProxyRouter
        # 原始 usage chunk 同时带 stale choices + finish_reason（应被丢弃，只留 usage 对象）
        usage_frame = {"id": "u", "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": {"content": "原始违规"},
                                    "finish_reason": "stop"}],
                       "usage": {"total_tokens": 42}}
        buffered = [
            {"choices": [{"delta": {"content": "原始违规"}, "index": 0}]},
            usage_frame,
        ]
        out = DeepProxyRouter._rebuild_stream_frames(
            {"role": "assistant", "content": "修正后"}, buffered,
        )
        usage_out = [f for f in out if f.get("usage")]
        assert len(usage_out) == 1
        assert usage_out[0]["usage"] == {"total_tokens": 42}
        assert usage_out[0]["choices"] == []  # stale choices 被剥离
        # 重建帧里不得泄漏原始违规正文
        leaked = "".join(
            c.get("delta", {}).get("content", "")
            for f in out for c in f.get("choices", [])
        )
        assert "原始违规" not in leaked
        # 只有一个终止 finish_reason
        finishes = [c.get("finish_reason") for f in out for c in f.get("choices", [])
                    if c.get("finish_reason")]
        assert finishes == ["stop"]
