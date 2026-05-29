"""client_stream 三单元单测（mock chunk 流 / awaitable，快）。"""
from __future__ import annotations

import asyncio

from deep_proxy.cross_consult.client_stream import with_heartbeat, _Done


async def test_with_heartbeat_emits_pings_then_result():
    async def slow():
        await asyncio.sleep(0.25)
        return "answer"

    frames = []
    result = None
    async for f in with_heartbeat(slow(), heartbeat_seconds=0.1):
        if isinstance(f, _Done):
            result = f.value
        else:
            frames.append(f)

    assert result == "answer"
    assert frames and all(fr == {"_dp_heartbeat": True} for fr in frames)
    assert len(frames) >= 2  # ~0.25s / 0.1s


async def test_with_heartbeat_fast_awaitable_no_ping():
    async def fast():
        return 42

    frames = [f async for f in with_heartbeat(fast(), heartbeat_seconds=5.0)]
    assert len(frames) == 1
    assert isinstance(frames[0], _Done) and frames[0].value == 42


from deep_proxy.cross_consult.client_stream import stream_one_turn, TurnResult


def _delta_chunk(**delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _finish_chunk(reason):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def _cc_tool_call_delta():
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "cc1", "type": "function",
         "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}
    ]}, "finish_reason": None}]}


async def _iter(chunks):
    for c in chunks:
        yield c


async def test_stream_one_turn_forwards_content_and_reasoning_live():
    chunks = [
        _delta_chunk(role="assistant"),
        _delta_chunk(reasoning_content="思考"),
        _delta_chunk(content="答案"),
        _finish_chunk("stop"),
    ]
    res = TurnResult()
    out = [f async for f in stream_one_turn(
        _iter(chunks), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    texts = [d["choices"][0]["delta"] for d in out if "choices" in d]
    assert {"reasoning_content": "思考"} in texts
    assert {"content": "答案"} in texts
    assert res.had_cc_call is False
    assert res.content == "答案"
    assert res.finish_reason == "stop"


async def test_stream_one_turn_suppresses_cc_tool_call_frames():
    chunks = [
        _delta_chunk(content="让我咨询"),
        _cc_tool_call_delta(),
        _finish_chunk("tool_calls"),
    ]
    res = TurnResult()
    out = [f async for f in stream_one_turn(
        _iter(chunks), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    assert any(d.get("choices", [{}])[0].get("delta") == {"content": "让我咨询"} for d in out)
    assert not any("tool_calls" in d.get("choices", [{}])[0].get("delta", {}) for d in out)
    assert not any(d.get("choices", [{}])[0].get("finish_reason") for d in out)
    assert res.had_cc_call is True
    assert res.accumulated_tool_calls
    assert res.accumulated_tool_calls[0]["function"]["name"] == "cross_consult"


async def test_stream_one_turn_emits_heartbeat_on_gap():
    async def slow_gen():
        yield _delta_chunk(content="a")
        await asyncio.sleep(0.25)
        yield _delta_chunk(content="b")
        yield _finish_chunk("stop")

    res = TurnResult()
    out = [f async for f in stream_one_turn(
        slow_gen(), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert any(f == {"_dp_heartbeat": True} for f in out)
    assert res.errored is False


async def test_stream_one_turn_errors_when_budget_exceeded():
    async def hang_gen():
        await asyncio.sleep(1.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    out = [f async for f in stream_one_turn(
        hang_gen(), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=0.2, heartbeat_seconds=0.1,
    )]
    assert res.errored is True
    assert any(f == {"_dp_heartbeat": True} for f in out)


async def test_stream_one_turn_forwards_error_frame():
    chunks = [_delta_chunk(content="x"), {"error": {"message": "boom"}}]
    res = TurnResult()
    out = [f async for f in stream_one_turn(
        _iter(chunks), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    assert any(d.get("error") for d in out)
    assert res.errored is True


from unittest.mock import patch

from deep_proxy.cross_consult.client_stream import stream_cross_consult_continuation
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.providers import Provider
from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator


def _cc_tool_call(call_id="cc1", question="q"):
    return {"index": 0, "id": call_id, "type": "function",
            "function": {"name": "cross_consult",
                         "arguments": '{"question":"%s"}' % question}}


async def test_continuation_streams_consult_heartbeat_and_resend():
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    cfg.cross_consult = CrossConsultConfig(
        enabled=True, pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    cfg.providers["mimo"] = Provider(
        name="mimo", api_base="https://x", api_key="t", litellm_prefix="openai/",
        flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
    )
    source = cfg.providers["deepseek"]
    acc = StreamingReasoningAccumulator(request_messages=[])

    # 重发轮的流式 chunk（无 cc 调用 -> 终轮）
    async def resend_iter(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "综合答案"},
                            "finish_reason": "stop"}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "use cc"}]}

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks",
               new=resend_iter):
        frames = [f async for f in stream_cross_consult_continuation(
            initial_tool_calls=[_cc_tool_call()],
            body=body, source_provider=source, config=cfg,
            cc_config=cfg.cross_consult, accumulator=acc,
        )]

    # 重发 content 逐帧透传
    assert any(fr.get("choices", [{}])[0].get("delta", {}).get("content") == "综合答案"
               for fr in frames)
    # tool_result 已注入消息历史
    assert any(m.get("role") == "tool" and "外部视角" in str(m.get("content"))
               for m in body["messages"])


async def test_stream_one_turn_cancels_pending_task_on_early_close():
    """消费者提前关闭生成器（客户端断连）时，in-flight __anext__ task 被取消，
    不泄漏、不留 pending task。"""
    started = asyncio.Event()
    cancelled = {"hit": False}

    async def slow_gen():
        yield _delta_chunk(content="first")
        started.set()
        try:
            await asyncio.sleep(10)  # 模拟上游下一 chunk 迟迟不来
        except asyncio.CancelledError:
            cancelled["hit"] = True
            raise
        yield _delta_chunk(content="never")

    res = TurnResult()
    agen = stream_one_turn(
        slow_gen(), res, tool_name="cross_consult",
        idle_timeout=30.0, first_chunk_timeout=30.0, heartbeat_seconds=30.0,
    )
    # 取首帧后立即关闭生成器（等价客户端断连）
    first = await agen.__anext__()
    assert first["choices"][0]["delta"] == {"content": "first"}
    await agen.aclose()
    # 让事件循环跑一拍，取消传播到底层 gen
    await asyncio.sleep(0.05)
    assert cancelled["hit"] is True
