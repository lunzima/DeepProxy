"""iter_chat_chunks 中流式 StyleGuard 路径的集成测试。"""
import pytest
from deep_proxy.config import ProxyConfig, FlashUpgradeConfig
from deep_proxy.router import DeepProxyRouter


def _make_router(enabled=True, max_retries=2):
    cfg = ProxyConfig(
        style_guard={"enabled": enabled, "max_retries": max_retries},
        flash_upgrade=FlashUpgradeConfig(enabled=False),
    )
    return DeepProxyRouter(cfg)


async def _collect(async_gen):
    result = []
    async for frame in async_gen:
        result.append(frame)
    return result


def _chunk(content, finish_reason=None):
    return {"choices": [{"index": 0, "delta": {"content": content},
                         "finish_reason": finish_reason}]}


def _heartbeat():
    return {"_dp_heartbeat": True}


def _mock_upstream_result(content, reasoning_content="", finish_reason="stop"):
    msg = {"content": content, "role": "assistant"}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    return {"choices": [{"index": 0, "message": msg,
                         "finish_reason": finish_reason}]}


def _patch_stream(router, frames, mock_call_litellm=None):
    """猴子补丁：替换 _iter_plain_chunks 为模拟帧生成器（含 accumulator.consume），
    同时替换 call_litellm 为 mock 上游。返回 (orig_plain, orig_call)。"""
    import deep_proxy.router as mod

    async def mock_sub_gen(self, body, provider, accumulator):
        for f in frames:
            accumulator.consume(f)
            yield f

    orig_plain = router._iter_plain_chunks
    router._iter_plain_chunks = mock_sub_gen.__get__(router, type(router))

    orig_call = mod.call_litellm
    if mock_call_litellm is not None:
        mod.call_litellm = mock_call_litellm
    return orig_plain, orig_call


class TestStreamingStyleGuardHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_yielded_immediately(self):
        """心跳帧应在 StyleGuard 启用时立即透传。"""
        router = _make_router()
        frames = [_heartbeat(), _chunk("你好"), _heartbeat(), _chunk("世界", "stop")]

        async def upstream(config, body, provider=None):
            return _mock_upstream_result("你好世界")

        orig_plain, orig_call = _patch_stream(router, frames, upstream)
        try:
            result = await _collect(
                router.iter_chat_chunks({"messages": [{"role": "user", "content": "t"}]})
            )
            assert len(result) == 4
            assert result[0].get("_dp_heartbeat") is True
        finally:
            router._iter_plain_chunks = orig_plain
            import deep_proxy.router as mod
            mod.call_litellm = orig_call


class TestStreamingStyleGuardClean:
    @pytest.mark.asyncio
    async def test_clean_response_no_retry(self):
        """无违规时应原样 yield 原始帧。"""
        router = _make_router()
        frames = [_chunk("干净的文字"), _chunk("，符合规范。", "stop")]
        call_count = 0

        async def counting_upstream(config, body, provider=None):
            nonlocal call_count
            call_count += 1
            return _mock_upstream_result("干净的文字，符合规范。")

        orig_plain, orig_call = _patch_stream(router, frames, counting_upstream)
        try:
            result = await _collect(
                router.iter_chat_chunks({"messages": [{"role": "user", "content": "t"}]})
            )
            assert len(result) == 2
            assert call_count == 0
        finally:
            router._iter_plain_chunks = orig_plain
            import deep_proxy.router as mod
            mod.call_litellm = orig_call


class TestStreamingStyleGuardCorrection:
    @pytest.mark.asyncio
    async def test_violation_triggers_correction(self):
        """违规时应触发重发，yield 修正后单帧。"""
        router = _make_router()
        frames = [_chunk("他坐在那里，没有动。", "stop")]
        call_count = 0

        async def upstream(config, body, provider=None):
            nonlocal call_count
            call_count += 1
            return _mock_upstream_result("他坐在那里，双手搁在桌面。")

        orig_plain, orig_call = _patch_stream(router, frames, upstream)
        try:
            result = await _collect(
                router.iter_chat_chunks({"messages": [{"role": "user", "content": "t"}]})
            )
            assert call_count == 1
            assert len(result) == 1
            assert result[0]["choices"][0]["delta"]["content"] == "他坐在那里，双手搁在桌面。"
        finally:
            router._iter_plain_chunks = orig_plain
            import deep_proxy.router as mod
            mod.call_litellm = orig_call

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved(self):
        """修正后应保留 reasoning_content。"""
        router = _make_router()
        frames = [_chunk("他坐在那里，没有动。", "stop")]

        async def upstream(config, body, provider=None):
            return _mock_upstream_result(
                "他坐在那里，双手搁在桌面。",
                reasoning_content="模型推理过程...",
            )

        orig_plain, orig_call = _patch_stream(router, frames, upstream)
        try:
            result = await _collect(
                router.iter_chat_chunks({"messages": [{"role": "user", "content": "t"}]})
            )
            assert len(result) == 1
            delta = result[0]["choices"][0]["delta"]
            assert "reasoning_content" in delta
            assert delta["reasoning_content"] == "模型推理过程..."
        finally:
            router._iter_plain_chunks = orig_plain
            import deep_proxy.router as mod
            mod.call_litellm = orig_call


class TestStreamingStyleGuardFallback:
    @pytest.mark.asyncio
    async def test_exception_falls_back(self):
        """StyleGuard 异常时应回退到原始帧。"""
        router = _make_router()
        frames = [_chunk("他坐在那里，没有动。", "stop")]

        async def failing_upstream(config, body, provider=None):
            raise RuntimeError("模拟网络超时")

        orig_plain, orig_call = _patch_stream(router, frames, failing_upstream)
        try:
            result = await _collect(
                router.iter_chat_chunks({"messages": [{"role": "user", "content": "t"}]})
            )
            assert len(result) == 1
            assert result[0]["choices"][0]["delta"]["content"] == "他坐在那里，没有动。"
        finally:
            router._iter_plain_chunks = orig_plain
            import deep_proxy.router as mod
            mod.call_litellm = orig_call

    @pytest.mark.asyncio
    async def test_body_messages_rollback(self):
        """异常时应删除 StyleGuard 追加的反馈消息。"""
        router = _make_router()
        frames = [_chunk("他坐在那里，没有动。", "stop")]

        async def failing_upstream(config, body, provider=None):
            raise RuntimeError("模拟故障")

        orig_plain, orig_call = _patch_stream(router, frames, failing_upstream)
        body = {"messages": [{"role": "user", "content": "t"}]}
        original_count = len(body["messages"])
        try:
            await _collect(router.iter_chat_chunks(body))
            assert len(body["messages"]) == original_count
        finally:
            router._iter_plain_chunks = orig_plain
            import deep_proxy.router as mod
            mod.call_litellm = orig_call


class TestStreamingStyleGuardDisabled:
    @pytest.mark.asyncio
    async def test_yielded_immediately_when_disabled(self):
        """禁用 StyleGuard 时应直接 yield 帧。"""
        router = _make_router(enabled=False)
        frames = [_chunk("测试"), _chunk("内容", "stop")]
        orig_plain, _ = _patch_stream(router, frames)
        try:
            result = await _collect(
                router.iter_chat_chunks({"messages": [{"role": "user", "content": "t"}]})
            )
            assert len(result) == 2
        finally:
            router._iter_plain_chunks = orig_plain
