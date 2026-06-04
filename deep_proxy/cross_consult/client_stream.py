"""客户端真流式：cross_consult 激活时逐 token 透传 + 抑制虚拟工具帧 + 心跳桥接。

三个流式单元：
  - with_heartbeat：包裹 consult await，期间周期 yield 心跳帧
  - stream_one_turn：消费单轮上游 chunk 流，content/reasoning 即时透传、
    tool_calls 累加到轮末判定（结果写入传入的 TurnResult）、间隙发心跳
  - stream_cross_consult_continuation：execute_cross_consult_loop 的流式变体

辅助：TurnResult（单轮累加结果容器）、make_terminal_frame（终轮帧构造）。

哨兵（模块级单例 dict，不透传客户端）：
  - _HEARTBEAT {"_dp_heartbeat": True}：协议层序列化成 SSE 注释帧 / Anthropic ping
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Generic, TypeVar

from ..utils import is_error_frame, merge_tool_call_deltas
from ..config import ProxyConfig
from ..compatibility.reasoning_handler import StreamingReasoningAccumulator
from ..litellm_client import iter_litellm_chunks
from ..providers import Provider
from .config import CrossConsultConfig
from .reasoning_idle import chunk_has_reasoning as _has_reasoning_content
from .reasoning_idle import compute_reasoning_idle
from .interceptor import (
    extract_cross_consult_tool_calls,
    resolve_consult_tool_call,
    build_initial_response_from_stream_tool_calls,
)

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
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
            if task in done:
                yield _Done(task.result())
                return
            yield _HEARTBEAT
    finally:
        # 消费者提前关闭生成器（客户端断连 → GeneratorExit）时，取消并 drain
        # 仍 in-flight 的 awaitable（如 execute_consult），避免 "Task was destroyed
        # but it is pending" 警告与浪费的上游调用。
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass


@dataclass
class TurnResult:
    accumulated_tool_calls: list[dict] = field(default_factory=list)
    content: str = ""            # 累加的 assistant 文本，供重发轮重建消息历史
    had_cc_call: bool = False
    finish_reason: str | None = None
    errored: bool = False
    # 超时专属元数据（区别于上游 error frame）：errored=True 且 timed_out=True 时，
    # stream_turn_with_retry 据 phase/seconds 决定收尾（pre-content 重试 / 硬错误帧）。
    # timed_out=False 的 errored 表示真实上游 error frame（已逐帧透传）。
    timed_out: bool = False
    timeout_phase: str | None = None    # "first_chunk" | "mid_stream"
    timeout_seconds: float | None = None


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
    """把一个 chunk 的 content / tool_calls / finish_reason 累加进 result，并据累加后的
    tool_calls 重算 result.had_cc_call（是否已出现 name==tool_name 的调用——continuation
    据此判定终轮）。"""
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


def make_terminal_frame(finish_reason: str | None, tool_calls: list[dict]) -> dict:
    """构造终轮 choice 帧：带 finish_reason（及可选非 cc tool_calls）。

    供多处终轮帧复用（iter_chat_chunks 的 no-cc-call 分支、stream_cross_consult_continuation
    的终轮判定与硬轮次上限退出），保证形状一致。
    """
    delta: dict[str, Any] = {}
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return {"choices": [{
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason or "stop",
    }]}


@dataclass
class _Timeout:
    """consume_with_heartbeat 的超时哨兵：携带 phase（first_chunk / mid_stream）与
    触发的预算秒数，供调用方写入 TurnResult 并据 phase 构造通知文案。"""
    phase: str
    seconds: float


def _resolve_idle(idle_ref):
    """解析 idle_timeout：float 直接返回，list 取 [0]（mutable ref 供调用方动态调整）。"""
    if isinstance(idle_ref, list) and idle_ref:
        return float(idle_ref[0])
    return float(idle_ref)


async def consume_with_heartbeat(
    chunk_iter: AsyncIterator[dict],
    *,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    log_label: str,
) -> AsyncGenerator[Any, None]:
    """共享的"心跳化 idle-timeout 上游消费器"：消费上游 chunk 流并产出三类元素——

      - 上游 chunk dict（原样）
      - `_HEARTBEAT`（等待间隙的心跳哨兵）
      - `_Timeout(phase, seconds)`（首 chunk 超 first_chunk_timeout / 相邻 chunk 间隔超
        idle_timeout；产出后即 return）

    StopAsyncIteration（上游自然结束）→ 静默 return。finally 负责 cancel 仍 in-flight 的
    `__anext__` task + drain + aclose 上游，确保连接确定性释放（不依赖 GC）。

    idle_timeout 可以是 float（固定值）或 list[float]（mutable ref）：调用方可把 list
    放入一个可变容器如 `idle_ref = [idle_timeout]` 传入，后续每循环通过 _resolve_idle
    取最新值——检测到 reasoning_content 后动态放大的调用方按需写 idle_ref[0] = new_value。

    本函数只承载 "持久化 __anext__ task + 心跳 tick + first/inter-chunk 预算切换 + 超时
    检测 + 清理" 这套并发骨架；**per-chunk 处理与超时收尾留给调用方**（stream_one_turn
    累加/抑制 cc 工具帧并由 router 发通知；stream_with_idle_timeout 原样透传 + finish 短路、
    超时仅写 result 标志即 detection-only，重试/硬错误策略由 stream_with_retry 据 result
    决定）。它不写 TurnResult——超时元数据由各调用方据 `_Timeout` 自行写入，
    使 finish-then-hang 等"超时但不算错"的收尾无副作用。

    **调用方契约**：须在自身 finally 里 `await gen.aclose()` 本生成器，以便任何退出路径
    （正常 return / 异常 / 消费者断连触发的 GeneratorExit）都能确定性触发这里的 finally
    清理——`async for` 自身不会自动 aclose 内层生成器。

    与 streaming.aggregate_stream_to_response 的相似是**刻意分叉**，勿合并：后者用
    asyncio.wait_for（每轮新 task、无心跳、无 waited 累加）内部聚合成 dict，数据流方向相反。
    """
    it = chunk_iter.__aiter__() if hasattr(chunk_iter, "__aiter__") else chunk_iter
    got_first = False
    # 持久化的 __anext__ task：跨心跳 tick 复用，仅在 chunk 真正到达后才重建，
    # 避免每次 timeout 重拉而丢弃 in-flight 读。
    task: asyncio.Future = asyncio.ensure_future(it.__anext__())
    waited = 0.0
    try:
        while True:
            # 每轮动态解析 idle_timeout（调用方可能通过 mutable ref 更新）
            _idle = _resolve_idle(idle_timeout)
            budget = _idle if got_first else first_chunk_timeout
            step = heartbeat_seconds
            if budget and budget > 0:  # budget<=0 表示禁用该阶段超时（永不 trip）
                step = min(heartbeat_seconds, max(0.0, budget - waited))
            done, _ = await asyncio.wait(
                {task}, timeout=step if step > 0 else heartbeat_seconds,
            )
            if task not in done:
                waited += step
                if budget and budget > 0 and waited >= budget:
                    phase = "first_chunk" if not got_first else "mid_stream"
                    logger.warning("%s %s timeout after %.1fs", log_label, phase, budget)
                    yield _Timeout(phase, budget)
                    return
                yield _HEARTBEAT
                continue
            try:
                chunk = task.result()
            except StopAsyncIteration:
                return
            got_first = True
            waited = 0.0
            task = asyncio.ensure_future(it.__anext__())
            yield chunk
    finally:
        # 取消仍 in-flight 的 __anext__ task 并 drain（避免 "Task was destroyed but it is
        # pending" 警告），再 aclose 上游异步生成器促其 finally 运行（关闭 httpx 流 /
        # 释放连接），不依赖 GC 的非确定性回收。
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except BaseException:
                pass


async def stream_one_turn(
    chunk_iter: AsyncIterator[dict],
    result: TurnResult,
    *,
    tool_name: str,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    reasoning_idle: float | None = None,
) -> AsyncGenerator[Any, None]:
    """消费单轮上游 chunk 流：content/reasoning 即时透传；tool_calls 累加（不透传）
    留到轮末判定；等待间隙发心跳；error frame / 超预算 -> result.errored=True 并终止。

    reasoning_content 自适应：首次检测到 reasoning_content token 时，将 idle 预算升级到
    reasoning_idle（深度思考 burst 间隙属正常，需比 content idle 更宽容）。reasoning_idle
    为 None 时退回与 first_chunk_timeout 同级（向后兼容旧调用）。

    超时仅写 result 元数据（errored/timed_out/phase/seconds）后 return——收尾策略由调用方
    （stream_turn_with_retry：重试 / 硬错误）据 result.timed_out 决定。
    并发骨架与清理委托给 consume_with_heartbeat。
    """
    reasoning_idle_val = (
        max(idle_timeout, reasoning_idle) if reasoning_idle is not None
        else compute_reasoning_idle(idle_timeout, first_chunk_timeout)
    )
    idle_ref = [idle_timeout]
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_ref, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_one_turn",
    )
    try:
        async for item in gen:
            if isinstance(item, _Timeout):
                result.errored = True
                result.timed_out = True
                result.timeout_phase = item.phase
                result.timeout_seconds = item.seconds
                return
            if item is _HEARTBEAT:
                yield item
                continue
            chunk = item
            if is_error_frame(chunk):
                result.errored = True
                yield chunk
                return
            # reasoning_content 自适应：首次看到深度思考 token 时升级 idle 预算
            if _has_reasoning_content(chunk) and idle_ref[0] < reasoning_idle_val:
                idle_ref[0] = reasoning_idle_val
                logger.debug(
                    "stream_one_turn reasoning seen, idle %.0f→%.0f",
                    idle_timeout, reasoning_idle_val,
                )
            _accumulate_turn(chunk, result, tool_name)
            fwd = _client_facing_chunk(chunk)
            if fwd is not None:
                yield fwd
    finally:
        # 见 consume_with_heartbeat 的调用方契约：任何退出路径都 aclose 底层消费器，
        # 触发其 finally 清理上游（async for 自身不会自动 aclose 内层生成器）。
        await gen.aclose()


async def stream_with_idle_timeout(
    chunk_iter: AsyncIterator[dict],
    *,
    result: TurnResult,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    reasoning_idle: float | None = None,
) -> AsyncGenerator[Any, None]:
    """普通（非 cross_consult）流式路径的 idle 超时**检测器**：逐 chunk **原样透传**
    （不抑制 tool_calls / finish_reason），等待间隙发心跳；超预算时仅设 result.errored/
    timed_out/timeout_phase/timeout_seconds 并 return——**不注入任何通知帧**。

    重试 vs 硬错误的**策略**由调用方（stream_with_retry）据 result 元数据决定：旧"注入
    '请重试' content + clean stop"对 agent 结构上不可能触发重试（clean stop = 成功轮），
    已废弃（见 docs/superpowers/specs/2026-06-04-mid-stream-timeout-retry-design.md）。

    reasoning_content 自适应：首次检测到 reasoning_content token 时，将 idle 预算升级到
    reasoning_idle（深度思考 burst 间隙属正常，需比 content idle 更宽容）。reasoning_idle
    为 None 时退回与 first_chunk_timeout 同级（向后兼容旧调用）。

    与 stream_one_turn 的区别：本函数原样透传（无 cross_consult 工具帧累加/抑制）。两者
    共享同一 consume_with_heartbeat 骨架，差异收敛到 per-chunk 处理与超时收尾两处。
    """
    reasoning_idle_val = (
        max(idle_timeout, reasoning_idle) if reasoning_idle is not None
        else compute_reasoning_idle(idle_timeout, first_chunk_timeout)
    )
    idle_ref = [idle_timeout]
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_ref, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_with_idle_timeout",
    )
    saw_finish = False  # 上游是否已发过 finish_reason（本轮逻辑上已收尾）
    try:
        async for item in gen:
            if isinstance(item, _Timeout):
                # 上游已发 finish_reason 却不收尾（finish-then-hang）：本轮逻辑上已正常
                # 结束，直接退出，不标超时（否则会触发误重试）。
                if saw_finish:
                    return
                result.errored = True
                result.timed_out = True
                result.timeout_phase = item.phase
                result.timeout_seconds = item.seconds
                return
            if item is _HEARTBEAT:
                yield item
                continue
            chunk = item
            # reasoning_content 自适应：首次看到深度思考 token 时升级 idle 预算
            if _has_reasoning_content(chunk) and idle_ref[0] < reasoning_idle_val:
                idle_ref[0] = reasoning_idle_val
                logger.debug(
                    "stream_with_idle_timeout reasoning seen, idle %.0f→%.0f",
                    idle_timeout, reasoning_idle_val,
                )
            if any(c.get("finish_reason") for c in (chunk.get("choices") or [])):
                saw_finish = True
            yield chunk  # 原样透传（含 error frame / tool_calls / finish_reason）
    finally:
        await gen.aclose()


def make_hard_error_frame(reason: str) -> dict:
    """构造**客户端可见**的硬错误帧：`{"error": {...}}`（无 choices → is_error_frame True）。

    本帧经协议层透传给客户端（data: {...} + [DONE]），使 SDK 抛错，而非误判成一次成功轮。
    替代旧的"content 通知 + clean stop"（后者对 agent 结构上不可能触发重试）。
    """
    return {"error": {"message": reason, "type": "timeout_error", "param": None, "code": 504}}


def _frame_has_visible_output(frame: dict) -> bool:
    """frame 是否含**客户端可见输出**（content 文本 / tool_calls）。reasoning_content /
    reasoning（深度思考）**不算**——它不是答案，pre-content 重发可让其无害重来。
    心跳帧 / error 帧（无 choices）返回 False。"""
    for ch in frame.get("choices") or []:
        delta = ch.get("delta") or {}
        if isinstance(delta.get("content"), str) and delta["content"]:
            return True
        if delta.get("tool_calls"):
            return True
    return False


async def stream_turn_with_retry(
    make_attempt: Callable[[TurnResult, float], AsyncIterator[dict]],
    *,
    max_total_seconds: float,
    on_result: Callable[[TurnResult], None] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[dict, None]:
    """通用 pre-content 重试 + 硬错误骨架（plain 与 cross_consult 共享）。

    make_attempt(turn, remaining) 产出**一次全新尝试**的帧流——调用方在其中接好
    turn-streamer（plain: stream_with_idle_timeout；cc: stream_one_turn）、全新上游、
    accumulator 重置/回滚、以及把 pre-content 预算钳到 remaining。逐帧透传，据收尾决策：

      - 非超时收尾（干净成功 / 真实 error frame 已透传）：调用 on_result(turn) 把**胜出轮**
        TurnResult 交还调用方，再 return。
      - 超时且**已提交可见输出**（committed）：post-content 不可续传 → 发硬错误帧、return。
      - 超时且总预算（max_total_seconds 墙钟）耗尽：发硬错误帧、return。
      - 超时且 pre-content 且预算未尽：发一个心跳（保持连接温热）后**重试**。

    committed 一经置位不复位——重试只可能发生在任何可见 token 之前；一旦 content 开始，
    下一次停顿即硬错误。now 可注入以便测试确定性控制预算。
    见 docs/superpowers/specs/2026-06-04-cross-consult-retry-design.md。
    """
    deadline = now() + max_total_seconds
    committed = False
    while True:
        remaining = deadline - now()
        if remaining <= 0:
            yield make_hard_error_frame(
                f"上游持续无响应，超过 {max_total_seconds:g}s 总预算，本轮中断。"
            )
            return
        turn = TurnResult()
        async for frame in make_attempt(turn, remaining):
            if _frame_has_visible_output(frame):
                committed = True
            yield frame
        if not turn.timed_out:
            # 干净成功（含 StopAsyncIteration）或真实 error frame 已透传 → 交还胜出轮
            if on_result is not None:
                on_result(turn)
            return
        if committed:
            yield make_hard_error_frame(
                "已输出部分内容后上游中断，不可续传，本轮中断。"
            )
            return
        # pre-content stall：保持连接温热后重试（预算耗尽由下轮 remaining<=0 兜底）
        yield _HEARTBEAT


async def stream_with_retry(
    make_upstream: Callable[[], AsyncIterator[dict]],
    *,
    idle_timeout: float,
    reasoning_idle: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    max_total_seconds: float,
    now: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[dict, None]:
    """plain（非 cross_consult）路径适配器：passthrough turn-streamer
    （stream_with_idle_timeout）+ pre-content 预算钳制，委托给 stream_turn_with_retry。

    总预算守护：把 pre-content 预算（first_chunk / reasoning_idle）钳到 remaining，使单次
    pre-content 挂死不冲过 deadline 一整个 first_chunk_timeout。content idle 不钳——已
    committed 的健康流不应因临近预算被截断（其停顿本就立即硬错误）。
    见 docs/superpowers/specs/2026-06-04-mid-stream-timeout-retry-design.md。
    """
    def make_attempt(turn: TurnResult, remaining: float) -> AsyncIterator[dict]:
        return stream_with_idle_timeout(
            make_upstream(), result=turn,
            idle_timeout=idle_timeout,
            reasoning_idle=min(reasoning_idle, remaining),
            first_chunk_timeout=min(first_chunk_timeout, remaining),
            heartbeat_seconds=heartbeat_seconds,
        )
    async for frame in stream_turn_with_retry(
        make_attempt, max_total_seconds=max_total_seconds, now=now,
    ):
        yield frame


async def stream_cross_consult_continuation(
    *,
    initial_tool_calls: list[dict],
    body: dict[str, Any],
    source_provider: Provider,
    config: ProxyConfig,
    cc_config: CrossConsultConfig,
    accumulator: StreamingReasoningAccumulator,
    initial_content: str = "",
) -> AsyncGenerator[Any, None]:
    """execute_cross_consult_loop 的流式变体：执行 consult（间隙发心跳）+ 重发
    （逐 chunk 透传）+ 跨轮循环。yield 客户端帧 / 心跳帧 / error 帧。

    initial_tool_calls：初始轮已累加的 tool_calls（含至少一个 cross_consult 调用）。
    initial_content：初始轮已累加的 assistant 文本（前导文本）。须随首条 assistant
        消息一并写入对话历史——否则模型在 tool_call 前说的前导文本会从重发上下文
        丢失，与非流式 execute_cross_consult_loop（直接 append 完整 message）行为分叉。

    终轮帧契约：除硬错误 / 真实 error 退出（已发 error frame）外，本生成器在返回前总会
    yield 一个带 finish_reason 的终轮 choice 帧（终轮判定 / 无 cc 调用 / 硬轮次上限三处统一）。

    与 interceptor.execute_cross_consult_loop 是**并行实现，按设计分叉**：本函数走客户端
    真流式（yield 帧），后者走非流式（返回 dict）。共享 resolve_consult_tool_call 与同一
    轮次/配额策略；改配额/轮次规则时两处同步。
    """
    target_name = cc_config.pair_for(source_provider.name)
    target_provider = config.providers.get(target_name) if target_name else None
    if target_provider is None:
        return  # 无对偶，无可继续（调用方已透传初始内容）

    idle = float(cc_config.call_timeout_seconds)
    first = float(cc_config.first_chunk_timeout_seconds)
    hb = float(cc_config.stream_heartbeat_seconds)

    turn_tool_calls = initial_tool_calls
    turn_content = initial_content
    call_count = 0
    max_turns = cc_config.max_calls_per_request * 2 + 1

    for _turn in range(max_turns):
        pseudo = build_initial_response_from_stream_tool_calls(turn_tool_calls)
        # 终轮判定：本轮无 cross_consult 调用 -> 调用方已/将透传，停止
        cc_calls = extract_cross_consult_tool_calls(pseudo, cc_config.tool_name)
        if not cc_calls:
            return

        # 追加 assistant 消息（含本轮 content + 全部 tool_calls）到历史
        body["messages"].append({
            "role": "assistant",
            "content": turn_content or None,
            "tool_calls": turn_tool_calls,
        })

        for tc in cc_calls:
            tool_text = None
            async for frame in with_heartbeat(
                resolve_consult_tool_call(
                    tc, call_count=call_count,
                    target_provider=target_provider, config=config, cc_config=cc_config,
                ),
                heartbeat_seconds=hb,
            ):
                if isinstance(frame, _Done):
                    tool_text, consumed = frame.value
                    if consumed:
                        call_count += 1
                else:
                    yield frame  # 心跳
            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": tool_text,
            })

        # 重发：流式逐 chunk 透传 + pre-content 重试；复用同一 accumulator 写缓存。
        sc = config.streaming
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult, remaining: float):
            accumulator.restore(snap)   # 丢弃失败重发的累加，保留更早轮次内容
            return stream_one_turn(
                iter_litellm_chunks(
                    config, body, _accumulator=accumulator, provider=source_provider,
                ),
                turn, tool_name=cc_config.tool_name,
                idle_timeout=idle,
                reasoning_idle=min(float(sc.reasoning_idle_timeout_seconds), remaining),
                first_chunk_timeout=min(first, remaining),
                heartbeat_seconds=hb,
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_total_seconds=float(sc.max_stream_total_seconds),
            on_result=lambda t: captured.__setitem__("turn", t),
        ):
            yield frame
        turn = captured.get("turn")
        if turn is None or turn.errored:
            # 硬错误（已发 error frame）或真实上游 error frame（已逐帧透传）→ 终止。
            return
        turn_tool_calls = turn.accumulated_tool_calls
        turn_content = turn.content
        # 终轮（无 cc 调用）：把本轮 finish_reason / 非 cc tool_calls 作为终结帧透传
        if not turn.had_cc_call:
            yield make_terminal_frame(turn.finish_reason, turn_tool_calls)
            return

    # 硬轮次上限：内容已逐 chunk 透传完毕，补一个 finish_reason=stop 终轮帧让客户端
    # 正常收尾（对齐 no-cc-call 退出路径；否则客户端只收到 [DONE] 而无终轮 choice）。
    logger.warning("cross_consult stream continuation hit hard turn limit (%d)", max_turns)
    yield make_terminal_frame("stop", turn_tool_calls)
