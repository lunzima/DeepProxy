"""Plain-path mid-stream timeout 重试循环 + 硬错误（无 cross_consult）单测。

设计见 docs/superpowers/specs/2026-06-04-mid-stream-timeout-retry-design.md。
"""
from __future__ import annotations

import asyncio

from deep_proxy.config import StreamingConfig
from deep_proxy.utils import is_error_frame


# ----------------------------------------------------------------------------
# 共享 chunk 构造助手
# ----------------------------------------------------------------------------
def _delta_chunk(**delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _finish_chunk(reason):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def _tool_call_delta():
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "t", "type": "function",
         "function": {"name": "foo", "arguments": "{}"}}]},
        "finish_reason": None}]}


async def _iter(chunks):
    for c in chunks:
        yield c


# ----------------------------------------------------------------------------
# Increment 1: 配置字段
# ----------------------------------------------------------------------------
def test_streaming_config_new_defaults():
    sc = StreamingConfig()
    assert sc.idle_timeout_seconds == 15           # 60 → 15（content-phase "retry harder"）
    assert sc.reasoning_idle_timeout_seconds == 45  # 新增：推理阶段较大窗口
    assert sc.first_chunk_timeout_seconds == 120    # 不变
    assert sc.max_stream_total_seconds == 600       # 新增：总墙钟预算
    assert sc.heartbeat_seconds == 10               # 不变


# ----------------------------------------------------------------------------
# Increment 2: stream_with_idle_timeout 改为 detection-only（不再注入通知/clean stop）
# ----------------------------------------------------------------------------
from deep_proxy.cross_consult.client_stream import (  # noqa: E402
    stream_with_idle_timeout, TurnResult,
)


async def test_idle_timeout_detection_only_no_notice():
    """超时只写 result 元数据 + 发心跳；不再注入 [DeepProxy] 通知文本，也不发 clean stop。"""
    async def hang_gen():
        await asyncio.sleep(10.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    out = [f async for f in stream_with_idle_timeout(
        hang_gen(), result=res,
        idle_timeout=5.0, reasoning_idle=5.0,
        first_chunk_timeout=0.2, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is True
    assert res.timeout_phase == "first_chunk"
    # 不注入通知文本
    assert not any(
        "[DeepProxy]" in (d.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
        for d in out if "choices" in d
    )
    # 不注入 clean finish_reason=stop
    assert not any(
        d.get("choices", [{}])[0].get("finish_reason") == "stop"
        for d in out if "choices" in d
    )
    # 心跳仍发出（保持连接温热）
    assert any(f == {"_dp_heartbeat": True} for f in out)


async def test_idle_timeout_reasoning_idle_tolerates_reasoning_gap():
    """检测到 reasoning_content 后，idle 预算升到 reasoning_idle：推理 token 之间
    超过 content idle 但小于 reasoning_idle 的停顿**不**触发超时。"""
    async def reasoning_then_pause():
        yield _delta_chunk(reasoning_content="思考")
        await asyncio.sleep(0.35)          # > content idle 0.15，< reasoning_idle 2.0
        yield _delta_chunk(content="答案")
        yield _finish_chunk("stop")

    res = TurnResult()
    out = [f async for f in stream_with_idle_timeout(
        reasoning_then_pause(), result=res,
        idle_timeout=0.15, reasoning_idle=2.0,
        first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is False
    assert any(
        (d.get("choices", [{}])[0].get("delta", {}) or {}).get("content") == "答案"
        for d in out if "choices" in d
    )


# ----------------------------------------------------------------------------
# Increment 3: stream_with_retry 重试循环 + make_hard_error_frame 硬错误帧
# ----------------------------------------------------------------------------
from deep_proxy.cross_consult.client_stream import (  # noqa: E402
    stream_with_retry, make_hard_error_frame,
)


def _content_of(out, value):
    return any(
        (d.get("choices", [{}])[0].get("delta", {}) or {}).get("content") == value
        for d in out if "choices" in d
    )


def _has_finish(out, reason):
    return any(
        d.get("choices", [{}])[0].get("finish_reason") == reason
        for d in out if "choices" in d
    )


def test_make_hard_error_frame_is_error_frame():
    f = make_hard_error_frame("boom reason")
    assert is_error_frame(f)
    assert f["error"]["type"] == "timeout_error"
    assert f["error"]["code"] == 504
    assert "boom reason" in f["error"]["message"]


async def test_retry_succeeds_after_pre_content_stall():
    """首次尝试推理/prefill 阶段挂死（pre-content）→ 重发；第二次输出 content + finish。
    客户端看到无缝成功轮（含心跳），无硬错误。"""
    calls = {"n": 0}

    def make_upstream():
        calls["n"] += 1
        if calls["n"] == 1:
            async def stall():
                await asyncio.sleep(10.0)
                yield _delta_chunk(content="x")
            return stall()
        async def ok():
            yield _delta_chunk(content="hi")
            yield _finish_chunk("stop")
        return ok()

    out = [f async for f in stream_with_retry(
        make_upstream,
        idle_timeout=5.0, reasoning_idle=5.0,
        first_chunk_timeout=0.2, heartbeat_seconds=0.1,
        max_total_seconds=600.0,
    )]
    assert calls["n"] == 2                       # 重发了一次
    assert any(f == {"_dp_heartbeat": True} for f in out)
    assert _content_of(out, "hi")
    assert _has_finish(out, "stop")
    assert not any(is_error_frame(f) for f in out)  # 成功路径无硬错误


async def test_hard_error_on_post_content_stall():
    """已输出可见 content 后停顿（committed）→ 不可续传，立即硬错误，不重发。"""
    calls = {"n": 0}

    def make_upstream():
        calls["n"] += 1
        async def gen():
            yield _delta_chunk(content="partial")
            await asyncio.sleep(10.0)            # 已发可见 content 后挂死
            yield _delta_chunk(content="never")
        return gen()

    out = [f async for f in stream_with_retry(
        make_upstream,
        idle_timeout=0.2, reasoning_idle=0.2,
        first_chunk_timeout=5.0, heartbeat_seconds=0.1,
        max_total_seconds=600.0,
    )]
    assert calls["n"] == 1                        # committed 后不重发
    assert _content_of(out, "partial")
    assert any(is_error_frame(f) for f in out)


async def test_hard_error_on_budget_exhaustion():
    """每次尝试都在 pre-content 阶段挂死，且总预算耗尽 → 硬错误（注入 fake clock 控制预算）。"""
    times = [0.0, 100.0, 100.0]

    def fake_now():
        return times.pop(0) if len(times) > 1 else times[0]

    calls = {"n": 0}

    def make_upstream():
        calls["n"] += 1
        async def stall():
            await asyncio.sleep(10.0)
            yield _delta_chunk(content="x")
        return stall()

    out = [f async for f in stream_with_retry(
        make_upstream,
        idle_timeout=5.0, reasoning_idle=5.0,
        first_chunk_timeout=0.2, heartbeat_seconds=0.1,
        max_total_seconds=50.0, now=fake_now,
    )]
    assert calls["n"] == 1                        # deadline 已过 → 首次尝试后即硬错误
    assert any(is_error_frame(f) for f in out)


async def test_real_upstream_error_forwarded_no_retry():
    """真实上游 error frame 原样透传、不重发（只有超时才驱动重试循环）。"""
    calls = {"n": 0}

    def make_upstream():
        calls["n"] += 1
        async def gen():
            yield {"error": {"message": "upstream boom", "type": "api_error"}}
        return gen()

    out = [f async for f in stream_with_retry(
        make_upstream,
        idle_timeout=5.0, reasoning_idle=5.0,
        first_chunk_timeout=5.0, heartbeat_seconds=0.1,
        max_total_seconds=600.0,
    )]
    assert calls["n"] == 1
    assert any(is_error_frame(f) for f in out)


async def test_clean_finish_succeeds_first_try():
    calls = {"n": 0}

    def make_upstream():
        calls["n"] += 1
        async def gen():
            yield _delta_chunk(content="hello")
            yield _finish_chunk("stop")
        return gen()

    out = [f async for f in stream_with_retry(
        make_upstream,
        idle_timeout=5.0, reasoning_idle=5.0,
        first_chunk_timeout=5.0, heartbeat_seconds=5.0,
        max_total_seconds=600.0,
    )]
    assert calls["n"] == 1
    assert not any(is_error_frame(f) for f in out)
    assert _has_finish(out, "stop")


# ----------------------------------------------------------------------------
# Increment 4: _iter_plain_chunks 接入 stream_with_retry（经 iter_chat_chunks 集成）
# ----------------------------------------------------------------------------
def _fake_body():
    return {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}


async def test_iter_chat_chunks_retries_then_succeeds(router, monkeypatch):
    """plain 路径：首次尝试 pre-content 挂死 → 代理重发 → 第二次成功。客户端见 content+finish，
    无 error frame；升格记账在干净完成时照常提交。"""
    calls = {"n": 0}

    def fake_iter(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        if calls["n"] == 1:
            async def stall():
                await asyncio.sleep(10.0)
                yield _delta_chunk(content="x")
            return stall()
        async def ok():
            yield _delta_chunk(content="hi")
            yield _finish_chunk("stop")
        return ok()

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    router.config.streaming.first_chunk_timeout_seconds = 1
    router.config.streaming.heartbeat_seconds = 1

    out = [f async for f in router.iter_chat_chunks(_fake_body())]
    assert calls["n"] == 2
    assert _content_of(out, "hi")
    assert _has_finish(out, "stop")
    assert not any(is_error_frame(f) for f in out)


async def test_iter_chat_chunks_post_content_stall_forwards_error_frame(router, monkeypatch):
    """plain 路径：已输出 content 后挂死 → 硬错误帧**透传给客户端**（不再被 STREAM_ERRORED
    吞掉静默成功）。"""
    def fake_iter(config, body, *, _accumulator=None, provider=None):
        async def gen():
            yield _delta_chunk(content="partial")
            await asyncio.sleep(10.0)
            yield _delta_chunk(content="never")
        return gen()

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    router.config.streaming.idle_timeout_seconds = 1
    router.config.streaming.reasoning_idle_timeout_seconds = 1
    router.config.streaming.heartbeat_seconds = 1

    out = [f async for f in router.iter_chat_chunks(_fake_body())]
    assert _content_of(out, "partial")
    assert any(is_error_frame(f) for f in out)
