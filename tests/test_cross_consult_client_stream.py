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
