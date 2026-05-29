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
