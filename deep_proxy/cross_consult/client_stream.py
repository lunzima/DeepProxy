"""客户端真流式：cross_consult 激活时逐 token 透传 + 抑制虚拟工具帧 + 心跳桥接。

三个流式单元：
  - with_heartbeat：包裹 consult await，期间周期 yield 心跳帧
  - stream_one_turn：消费单轮上游 chunk 流，content/reasoning 即时透传、
    tool_calls 累加到轮末判定（结果写入传入的 TurnResult）、间隙发心跳
  - stream_cross_consult_continuation：execute_cross_consult_loop 的流式变体

辅助：TurnResult（单轮累加结果容器）、make_terminal_frame（终轮帧构造）。

哨兵（均为模块级单例 dict，不透传客户端）：
  - _HEARTBEAT {"_dp_heartbeat": True}：协议层序列化成 SSE 注释帧 / Anthropic ping
  - STREAM_ERRORED {"_dp_stream_errored": True}：通知 iter_chat_chunks 重发轮 errored
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Generic, TypeVar

from ..utils import is_error_frame, merge_tool_call_deltas
from ..config import ProxyConfig
from ..compatibility.reasoning_handler import StreamingReasoningAccumulator
from ..litellm_client import iter_litellm_chunks
from ..providers import Provider
from .config import CrossConsultConfig
from .interceptor import (
    extract_cross_consult_tool_calls,
    resolve_consult_tool_call,
    build_initial_response_from_stream_tool_calls,
)

logger = logging.getLogger(__name__)

_HEARTBEAT: dict[str, Any] = {"_dp_heartbeat": True}

# 跨模块哨兵（单例）：continuation 在重发轮 errored（超时/error）退出时 yield，供
# iter_chat_chunks 按 identity 识别并设 saw_error_frame（不提交升格记账）。不透传给客户端。
STREAM_ERRORED: dict[str, Any] = {"_dp_stream_errored": True}

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
    # 调用方据 phase/seconds 构造优雅超时通知（make_timeout_notice_frames），
    # 而非静默返回空轮。timed_out=False 的 errored 表示真实上游 error frame。
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
    的终轮判定与硬轮次上限退出、make_timeout_notice_frames 的收尾帧），保证形状一致。
    """
    delta: dict[str, Any] = {}
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return {"choices": [{
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason or "stop",
    }]}


def _timeout_notice_text(phase: str | None, seconds: float | None) -> str:
    """超时优雅通知正文。明确告知主 agent 这是**超时而非错误**、可重试，区分
    首 chunk 未达（疑似上游繁忙 / prefill 慢）与 mid-stream 停顿（输出中途断流）。"""
    secs = f"{seconds:g}s" if seconds and seconds > 0 else "超时窗口"
    if phase == "mid_stream":
        return (
            f"[DeepProxy] 上游在输出过程中连续 {secs} 没有新内容，本轮已中断。"
            "这不是错误，也不是最终答案——上游可能仍在繁忙或网络缓慢。"
            "请直接重试本次请求以继续。"
        )
    return (
        f"[DeepProxy] 上游在 {secs} 内未返回任何响应（疑似上游繁忙或网络缓慢），本轮已中断。"
        "这不是错误，也不是最终答案。请直接重试本次请求以继续。"
    )


def make_timeout_notice_frames(result: TurnResult) -> list[dict]:
    """据 TurnResult 的超时元数据构造**优雅通知帧**：一条 assistant content delta
    （告知主 agent 上游超时、可重试）+ 一个 finish_reason=stop 终轮帧。

    刻意走 content + clean finish（而非 error frame / HTTP 错误码）：让客户端把它当
    一次普通完成的 turn 收尾——agent 读到通知文本后自行决定重试，而不是收到空轮后
    静默停止推理（见本模块修复的根因），也不会因错误码中断会话。
    """
    notice = _timeout_notice_text(result.timeout_phase, result.timeout_seconds)
    return [
        {"choices": [{"index": 0, "delta": {"content": notice}, "finish_reason": None}]},
        make_terminal_frame("stop", []),
    ]


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
) -> AsyncGenerator[Any, None]:
    """共享的"心跳化 idle-timeout 上游消费器"：消费上游 chunk 流并产出三类元素——

      - 上游 chunk dict（原样）
      - `_HEARTBEAT`（等待间隙的心跳哨兵）
      - `_Timeout(phase, seconds)`（首 chunk 超 first_chunk_timeout / 相邻 chunk 间隔超
        idle_timeout；产出后即 return）

    StopAsyncIteration（上游自然结束）→ 静默 return。finally 负责 cancel 仍 in-flight 的
    `__anext__` task + drain + aclose 上游，确保连接确定性释放（不依赖 GC）。

    本函数只承载 "持久化 __anext__ task + 心跳 tick + first/inter-chunk 预算切换 + 超时
    检测 + 清理" 这套并发骨架；**per-chunk 处理与超时收尾留给调用方**（stream_one_turn
    累加/抑制 cc 工具帧并由 router 发通知；stream_with_idle_timeout 原样透传 + finish 短路
    并内联发通知）。它不写 TurnResult——超时元数据由各调用方据 `_Timeout` 自行写入，
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
            budget = idle_timeout if got_first else first_chunk_timeout
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
) -> AsyncGenerator[Any, None]:
    """消费单轮上游 chunk 流：content/reasoning 即时透传；tool_calls 累加（不透传）
    留到轮末判定；等待间隙发心跳；error frame / 超预算 -> result.errored=True 并终止。

    超时仅写 result 元数据（errored/timed_out/phase/seconds）后 return——通知帧由调用方
    （router.iter_chat_chunks 初始轮 / continuation 重发轮）据 result.timed_out 构造。
    并发骨架与清理委托给 consume_with_heartbeat。
    """
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
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
) -> AsyncGenerator[Any, None]:
    """普通（非 cross_consult）流式路径的 idle 超时守护：逐 chunk **原样透传**
    （不抑制 tool_calls / finish_reason），等待间隙发心跳；超预算时设 result.errored/
    timed_out 并内联 yield 优雅超时通知（content + finish_reason=stop）并终止。

    与 stream_one_turn 的区别：本函数原样透传（无 cross_consult 工具帧累加/抑制），且
    超时通知由本函数内联发出（普通路径无 router 侧的 continuation 编排）。两者共享同一
    consume_with_heartbeat 骨架，差异收敛到 per-chunk 处理与超时收尾两处。
    """
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_with_idle_timeout",
    )
    saw_finish = False  # 上游是否已发过 finish_reason（本轮逻辑上已收尾）
    try:
        async for item in gen:
            if isinstance(item, _Timeout):
                # 上游已发 finish_reason 却不收尾（finish-then-hang）：本轮逻辑上已正常
                # 结束，直接退出，**不**再注入通知/第二个 finish_reason（否则一条流出现
                # 两个 finish，违反协议）。
                if saw_finish:
                    return
                result.errored = True
                result.timed_out = True
                result.timeout_phase = item.phase
                result.timeout_seconds = item.seconds
                for frame in make_timeout_notice_frames(result):
                    yield frame
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

    终轮帧契约：除 error 退出（yield STREAM_ERRORED）外，本生成器在返回前总会 yield
    一个带 finish_reason 的终轮 choice 帧（终轮判定 / 无 cc 调用 / 硬轮次上限三处统一）。

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

        # 重发：流式，逐 chunk 透传；复用同一 accumulator 写缓存
        resend_iter = iter_litellm_chunks(
            config, body, _accumulator=accumulator, provider=source_provider,
        )
        turn = TurnResult()
        async for frame in stream_one_turn(
            resend_iter, turn, tool_name=cc_config.tool_name,
            idle_timeout=idle, first_chunk_timeout=first, heartbeat_seconds=hb,
        ):
            yield frame
        if turn.errored:
            # 重发轮超时：先 yield 优雅超时通知（content + finish_reason=stop），让 agent
            # 读到"上游超时、可重试"而非静默收到空轮后停止推理（根因修复）。真实上游 error
            # frame（timed_out=False）已在上面的 stream_one_turn 循环里逐帧透传，不重复发。
            if turn.timed_out:
                for frame in make_timeout_notice_frames(turn):
                    yield frame
            # STREAM_ERRORED：通知调用方设 saw_error_frame（不提交升格记账），
            # 该哨兵被 iter_chat_chunks 吞掉，不透传给客户端。
            yield STREAM_ERRORED
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
