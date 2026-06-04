"""流式超时引擎 + 重试骨架 + cross_consult 真流式（逐 token 透传 + 抑制虚拟工具帧 + 心跳）。

**单一超时引擎**：
  - consume_with_heartbeat：唯一的"心跳化 idle-timeout 上游消费器"，自持 first-chunk /
    content idle / reasoning-aware idle 升级；产出 chunk / _HEARTBEAT / _Timeout。plain、
    cross_consult、aggregate_stream_to_response 三方共用（streaming.py 驱动它聚合成 dict）。

**重试骨架（count-based）**：
  - stream_turn_with_retry：通用 pre-content 重试（最多 max_retries 次）+ post-content/
    耗尽硬错误帧；健康流（持续产出）永不被打断。
  - stream_with_retry：plain 路径适配器（passthrough turn-streamer）。

**消费者（差异仅在 per-chunk 处理）**：
  - stream_with_idle_timeout：plain 原样透传（detection-only：超时只写 TurnResult）。
  - stream_one_turn：cross_consult 单轮——content/reasoning 即时透传、cc 工具帧累加/抑制。
  - stream_cross_consult_continuation：consult 执行（with_heartbeat 桥接心跳）+ 重发循环。

辅助：TurnResult（单轮累加容器）、make_terminal_frame（终轮帧）、make_hard_error_frame
（客户端可见硬错误帧）、_frame_has_visible_output（committed 判定）。

哨兵（模块级单例 dict，不透传客户端）：
  - _HEARTBEAT {"_dp_heartbeat": True}：协议层序列化成 SSE 注释帧 / Anthropic ping
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Generic, TypeVar

from ..utils import is_error_frame, merge_tool_call_deltas
from ..config import ProxyConfig
from ..compatibility.reasoning_handler import StreamingReasoningAccumulator
from ..litellm_client import iter_litellm_chunks
from ..providers import Provider
from .config import CrossConsultConfig
from .reasoning_idle import chunk_has_reasoning as _has_reasoning_content
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


async def consume_with_heartbeat(
    chunk_iter: AsyncIterator[dict],
    *,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    log_label: str,
    reasoning_idle: float | None = None,
) -> AsyncGenerator[Any, None]:
    """**单一超时引擎**：消费上游 chunk 流并产出三类元素——

      - 上游 chunk dict（原样）
      - `_HEARTBEAT`（等待间隙的心跳哨兵）
      - `_Timeout(phase, seconds)`（首 chunk 超 first_chunk_timeout / 相邻 chunk 间隔超
        idle；产出后即 return）

    自持有全部超时与 reasoning 自适应逻辑：
      - 首 chunk 前用 first_chunk_timeout；其后用 idle（content 阶段）。
      - 首次见到非空 reasoning_content 后，idle 升到 max(idle, reasoning_idle)（深度思考
        burst 间隙属正常）。reasoning_idle=None → 不升级（content idle 全程）。
      - budget<=0 表示禁用该阶段超时（永不 trip）。
    **健康流（持续产出 token）永不被打断**：每个 chunk 到达即把 waited 归零。

    StopAsyncIteration（上游自然结束）→ 静默 return。finally 负责 cancel 仍 in-flight 的
    `__anext__` task + drain + aclose 上游，确保连接确定性释放（不依赖 GC）。

    本引擎只承载 "持久化 __anext__ task + 心跳 tick + first/idle 预算切换 + reasoning 升级 +
    超时检测 + 清理"；**per-chunk 处理与收尾留给消费者**（stream_with_idle_timeout 透传、
    stream_one_turn 累加/抑制 cc 帧、aggregate_stream_to_response 聚合成 dict）。它不写
    TurnResult——超时元数据由消费者据 `_Timeout` 自行处理。

    **调用方契约**：须在自身 finally 里 `await gen.aclose()` 本生成器，以便任何退出路径
    都能确定性触发这里的 finally 清理（`async for` 自身不会 aclose 内层生成器）。
    """
    it = chunk_iter.__aiter__() if hasattr(chunk_iter, "__aiter__") else chunk_iter
    got_first = False
    current_idle = idle_timeout  # content 阶段 idle；见到 reasoning 后升到 reasoning_idle
    # 持久化的 __anext__ task：跨心跳 tick 复用，仅在 chunk 真正到达后才重建，
    # 避免每次 timeout 重拉而丢弃 in-flight 读。
    task: asyncio.Future = asyncio.ensure_future(it.__anext__())
    waited = 0.0
    try:
        while True:
            budget = current_idle if got_first else first_chunk_timeout
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
            # reasoning 自适应：首见深度思考 token 后升级 idle 预算
            if (reasoning_idle is not None and current_idle < reasoning_idle
                    and _has_reasoning_content(chunk)):
                current_idle = max(current_idle, reasoning_idle)
                logger.debug("%s reasoning seen, idle→%.0f", log_label, current_idle)
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

    reasoning_content 自适应、first/idle 预算与清理全部由 consume_with_heartbeat 负责
    （reasoning_idle=None 时不升级，content idle 全程）。

    超时仅写 result 元数据（errored/timed_out/phase/seconds）后 return——收尾策略由调用方
    （stream_turn_with_retry：重试 / 硬错误）据 result.timed_out 决定。
    超时检测 + reasoning 自适应 + 清理全部委托给 consume_with_heartbeat；本函数只做
    per-chunk 累加/抑制。
    """
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_one_turn",
        reasoning_idle=reasoning_idle,
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

    reasoning_content 自适应、first/idle 预算与清理全部由 consume_with_heartbeat 负责
    （reasoning_idle=None 时不升级，content idle 全程）。

    与 stream_one_turn 的区别：本函数原样透传（无 cross_consult 工具帧累加/抑制）。两者
    共享同一 consume_with_heartbeat 引擎，差异仅在 per-chunk 处理与超时收尾。
    """
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_with_idle_timeout",
        reasoning_idle=reasoning_idle,
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
    make_attempt: Callable[[TurnResult], AsyncIterator[dict]],
    *,
    max_retries: int,
    on_result: Callable[[TurnResult], None] | None = None,
) -> AsyncGenerator[dict, None]:
    """通用 pre-content 重试 + 硬错误骨架（plain 与 cross_consult 共享）。

    make_attempt(turn) 产出**一次全新尝试**的帧流——调用方在其中接好 turn-streamer
    （plain: stream_with_idle_timeout；cc: stream_one_turn）、全新上游、accumulator 重置/
    回滚。每尝试用**自然** idle/first_chunk 窗口（不钳制）——健康流（持续产出）永不被打断，
    只有 dead-air 才 stall。据收尾决策：

      - 非超时收尾（干净成功 / 真实 error frame 已透传）：调用 on_result(turn) 把**胜出轮**
        TurnResult 交还调用方，再 return。
      - 超时且**已提交可见输出**（committed）：post-content 不可续传 → 发硬错误帧、return。
      - 超时且 pre-content 且重发次数已达 max_retries：发硬错误帧、return。
      - 超时且 pre-content 且还可重发：发心跳后重发（retries +1）。

    committed 一经置位不复位——重试只可能发生在任何可见 token 之前。
    见 docs/superpowers/plans/2026-06-04-timeout-simplification.md。
    """
    committed = False
    retries = 0
    while True:
        turn = TurnResult()
        async for frame in make_attempt(turn):
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
        if retries >= max_retries:
            yield make_hard_error_frame(
                f"上游持续无响应，已重发 {retries} 次仍未恢复，本轮中断。"
            )
            return
        retries += 1
        # pre-content stall：保持连接温热后重发原请求
        yield _HEARTBEAT


async def stream_with_retry(
    make_upstream: Callable[[], AsyncIterator[dict]],
    *,
    idle_timeout: float,
    reasoning_idle: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    max_retries: int,
) -> AsyncGenerator[dict, None]:
    """plain（非 cross_consult）路径适配器：passthrough turn-streamer
    （stream_with_idle_timeout）+ 自然超时窗口（不钳制），委托给 stream_turn_with_retry。"""
    def make_attempt(turn: TurnResult) -> AsyncIterator[dict]:
        return stream_with_idle_timeout(
            make_upstream(), result=turn,
            idle_timeout=idle_timeout, reasoning_idle=reasoning_idle,
            first_chunk_timeout=first_chunk_timeout, heartbeat_seconds=heartbeat_seconds,
        )
    async for frame in stream_turn_with_retry(make_attempt, max_retries=max_retries):
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

    sc = config.streaming
    hb = float(sc.heartbeat_seconds)   # consult 等待间隙心跳

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

        # 重发：流式逐 chunk 透传 + pre-content 重试（自然窗口、不钳制）；复用同一 accumulator。
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult):
            accumulator.restore(snap)   # 丢弃失败重发的累加，保留更早轮次内容
            return stream_one_turn(
                iter_litellm_chunks(
                    config, body, _accumulator=accumulator, provider=source_provider,
                ),
                turn, tool_name=cc_config.tool_name,
                idle_timeout=float(sc.idle_timeout_seconds),
                reasoning_idle=float(sc.reasoning_idle_timeout_seconds),
                first_chunk_timeout=float(sc.first_chunk_timeout_seconds),
                heartbeat_seconds=float(sc.heartbeat_seconds),
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_retries=int(sc.max_retries),
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
