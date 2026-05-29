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

from ..utils import merge_tool_call_deltas

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


@dataclass
class TurnResult:
    accumulated_tool_calls: list[dict] = field(default_factory=list)
    content: str = ""            # 累加的 assistant 文本，供重发轮重建消息历史
    had_cc_call: bool = False
    finish_reason: str | None = None
    errored: bool = False


def _client_facing_chunk(chunk: dict) -> dict | None:
    """从上游 chunk 构造仅含 content/reasoning 的客户端帧（剥 tool_calls、
    抑制 finish_reason）。无可透传内容时返回 None。"""
    out_choices = []
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        fwd: dict[str, Any] = {}
        if delta.get("role"):
            fwd["role"] = delta["role"]
        if isinstance(delta.get("content"), str):
            fwd["content"] = delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            fwd["reasoning_content"] = delta["reasoning_content"]
        if isinstance(delta.get("reasoning"), str):
            fwd["reasoning"] = delta["reasoning"]
        # 仅 role（无 content/reasoning）的空壳不值得单独发
        if not fwd or set(fwd.keys()) == {"role"}:
            continue
        out_choices.append({"index": ch.get("index", 0), "delta": fwd,
                            "finish_reason": None})
    if not out_choices:
        return None
    return {"choices": out_choices}


def _accumulate_turn(chunk: dict, result: TurnResult, tool_name: str) -> None:
    """把一个 chunk 的 tool_calls / content / finish_reason 累加进 result。"""
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        if isinstance(delta.get("content"), str):
            result.content += delta["content"]
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            result.accumulated_tool_calls = merge_tool_call_deltas(
                result.accumulated_tool_calls, tcs,
            )
        fr = ch.get("finish_reason")
        if fr:
            result.finish_reason = fr
    result.had_cc_call = any(
        (tc.get("function") or {}).get("name") == tool_name
        for tc in result.accumulated_tool_calls
    )


async def stream_one_turn(
    chunk_iter: AsyncIterator[dict],
    result: TurnResult,
    *,
    tool_name: str,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
) -> AsyncGenerator[Any, None]:
    """消费单轮上游 chunk 流：content/reasoning 即时透传；tool_calls 累加（不透传）
    留到轮末判定；等待间隙发心跳；error frame / 超预算 -> result.errored=True 并终止。

    心跳/预算：等待下一 chunk 时每 heartbeat_seconds 无 chunk 发一次心跳；累计等待
    超过预算（首 chunk 用 first_chunk_timeout，之后 idle_timeout）视为 hang。
    """
    it = chunk_iter.__aiter__() if hasattr(chunk_iter, "__aiter__") else chunk_iter
    got_first = False
    task: asyncio.Future | None = asyncio.ensure_future(it.__anext__())
    waited = 0.0
    while True:
        budget = idle_timeout if got_first else first_chunk_timeout
        step = heartbeat_seconds
        if budget and budget > 0:
            step = min(heartbeat_seconds, max(0.0, budget - waited))
        done, _ = await asyncio.wait({task}, timeout=step if step > 0 else heartbeat_seconds)
        if task not in done:
            waited += step
            if budget and budget > 0 and waited >= budget:
                logger.warning(
                    "stream_one_turn %s timeout after %.1fs",
                    "first_chunk" if not got_first else "mid_stream", budget,
                )
                result.errored = True
                task.cancel()
                return
            yield _HEARTBEAT
            continue
        # chunk 到达
        try:
            chunk = task.result()
        except StopAsyncIteration:
            return
        got_first = True
        waited = 0.0
        task = asyncio.ensure_future(it.__anext__())

        if isinstance(chunk.get("error"), dict) and not chunk.get("choices"):
            result.errored = True
            yield chunk
            task.cancel()
            return

        _accumulate_turn(chunk, result, tool_name)
        fwd = _client_facing_chunk(chunk)
        if fwd is not None:
            yield fwd
