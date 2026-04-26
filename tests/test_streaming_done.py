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
