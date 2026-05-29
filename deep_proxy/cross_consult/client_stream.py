"""客户端真流式：cross_consult 激活时逐 token 透传 + 抑制虚拟工具帧 + 心跳桥接。

三单元：
  - with_heartbeat：包裹 consult await，期间周期 yield 心跳帧
  - stream_one_turn：消费单轮上游 chunk 流，content/reasoning 即时透传、
    tool_calls 累加到轮末判定、间隙发心跳
  - stream_cross_consult_continuation：execute_cross_consult_loop 的流式变体

心跳 sentinel = {"_dp_heartbeat": True}（dict），由协议层序列化成 SSE 注释帧。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Generic, TypeVar

logger = logging.getLogger(__name__)

_HEARTBEAT: dict[str, Any] = {"_dp_heartbeat": True}

T = TypeVar("T")


@dataclass
class _Done(Generic[T]):
    """with_heartbeat 的终结哨兵：携带被包裹 awaitable 的结果。"""
    value: T


async def with_heartbeat(
    awaitable: Awaitable[T], *, heartbeat_seconds: float,
) -> AsyncGenerator[Any, None]:
    """运行 awaitable，期间每 heartbeat_seconds 无完成就 yield 一个心跳帧；
    完成后 yield 单个 _Done(result)。"""
    task = asyncio.ensure_future(awaitable)
    while True:
        done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
        if task in done:
            yield _Done(task.result())
            return
        yield _HEARTBEAT
