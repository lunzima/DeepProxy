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


from deep_proxy.cross_consult.client_stream import (
    stream_one_turn, TurnResult, make_timeout_notice_frames, stream_with_idle_timeout,
)


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


async def test_stream_one_turn_records_first_chunk_timeout_metadata():
    """超时（区别于上游 error frame）须在 result 上留下足够元数据，供调用方
    构造优雅通知（phase + budget），而非静默返回空轮。"""
    async def hang_gen():
        await asyncio.sleep(1.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    _ = [f async for f in stream_one_turn(
        hang_gen(), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=0.2, heartbeat_seconds=0.1,
    )]
    assert res.errored is True
    assert res.timed_out is True
    assert res.timeout_phase == "first_chunk"
    assert res.timeout_seconds == 0.2


async def test_stream_one_turn_records_mid_stream_timeout_metadata():
    async def slow_gen():
        yield _delta_chunk(content="a")
        await asyncio.sleep(1.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    _ = [f async for f in stream_one_turn(
        slow_gen(), res, tool_name="cross_consult",
        idle_timeout=0.2, first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is True
    assert res.timeout_phase == "mid_stream"
    assert res.timeout_seconds == 0.2


async def test_stream_one_turn_reasoning_upgrades_idle_budget():
    """检测到 reasoning_content 后，mid-stream idle 预算升级到 max(idle, first_chunk)。
    深度思考 burst 间隙 > idle_timeout 但 ≤ reasoning_idle 时不应误判为 hang。
    （回归：此用例在 162f3d9 之前会 mid_stream 超时。）"""
    async def reasoning_then_gap():
        yield _delta_chunk(reasoning_content="思考中…")  # 触发升格 max(0.2, 2.0)=2.0
        await asyncio.sleep(0.5)  # > idle_timeout(0.2)，< reasoning_idle(2.0)
        yield _delta_chunk(content="答案")
        yield _finish_chunk("stop")

    res = TurnResult()
    out = [f async for f in stream_one_turn(
        reasoning_then_gap(), res, tool_name="cross_consult",
        idle_timeout=0.2, first_chunk_timeout=2.0, heartbeat_seconds=0.1,
    )]
    assert res.errored is False
    assert res.timed_out is False
    texts = [d["choices"][0]["delta"] for d in out if "choices" in d]
    assert {"reasoning_content": "思考中…"} in texts
    assert {"content": "答案"} in texts


def test_make_timeout_notice_frames_first_chunk():
    res = TurnResult(timed_out=True, timeout_phase="first_chunk", timeout_seconds=120.0)
    frames = make_timeout_notice_frames(res)
    assert len(frames) == 2
    content = frames[0]["choices"][0]["delta"]["content"]
    assert "[DeepProxy]" in content and "120" in content
    assert frames[0]["choices"][0]["finish_reason"] is None
    # 终轮帧：clean finish（非 error），让 agent 当普通一轮收尾
    assert frames[1]["choices"][0]["finish_reason"] == "stop"


def test_make_timeout_notice_frames_mid_stream_distinct_text():
    first = make_timeout_notice_frames(
        TurnResult(timed_out=True, timeout_phase="first_chunk", timeout_seconds=10.0))
    mid = make_timeout_notice_frames(
        TurnResult(timed_out=True, timeout_phase="mid_stream", timeout_seconds=10.0))
    first_text = first[0]["choices"][0]["delta"]["content"]
    mid_text = mid[0]["choices"][0]["delta"]["content"]
    assert first_text != mid_text
    # mid_stream 语义关键词 + 前缀 + 秒数（避免仅"不相等"的弱断言）
    assert "[DeepProxy]" in mid_text
    assert "输出过程中" in mid_text and "10" in mid_text
    assert mid[1]["choices"][0]["finish_reason"] == "stop"


async def test_stream_with_idle_timeout_forwards_chunks_verbatim():
    """普通（非 cc）流式路径：原样透传，不抑制 tool_calls / finish_reason。"""
    chunks = [
        _delta_chunk(content="a"),
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "t", "type": "function",
             "function": {"name": "foo", "arguments": "{}"}}]},
            "finish_reason": None}]},
        _finish_chunk("tool_calls"),
    ]
    res = TurnResult()
    out = [f async for f in stream_with_idle_timeout(
        _iter(chunks), result=res,
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    assert any("tool_calls" in d.get("choices", [{}])[0].get("delta", {}) for d in out)
    assert any(d.get("choices", [{}])[0].get("finish_reason") == "tool_calls" for d in out)
    assert res.timed_out is False
    assert res.errored is False


async def test_stream_with_idle_timeout_emits_notice_on_timeout():
    async def hang_gen():
        await asyncio.sleep(1.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    out = [f async for f in stream_with_idle_timeout(
        hang_gen(), result=res,
        idle_timeout=5.0, first_chunk_timeout=0.2, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is True
    assert any("[DeepProxy]" in d.get("choices", [{}])[0].get("delta", {}).get("content", "")
               for d in out)
    assert any(d.get("choices", [{}])[0].get("finish_reason") == "stop" for d in out)
    # 心跳在等待间隙发出（保持连接温热）
    assert any(f == {"_dp_heartbeat": True} for f in out)


async def test_stream_with_idle_timeout_no_double_finish_after_upstream_finish():
    """上游已发 finish_reason 后却不收尾（hang，不抛 StopAsyncIteration）：本轮逻辑上
    已正常结束,idle 触发时**不得**再注入超时通知/第二个 finish_reason（否则一条流出现
    两个 finish，违反协议）。"""
    started = asyncio.Event()

    async def finish_then_hang():
        yield _delta_chunk(content="done")
        yield _finish_chunk("stop")
        started.set()
        await asyncio.Event().wait()  # 发完 finish 后永久挂起（不抛 StopAsyncIteration）
        yield _delta_chunk(content="never")

    res = TurnResult()
    out = [f async for f in stream_with_idle_timeout(
        finish_then_hang(), result=res,
        idle_timeout=0.2, first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    finishes = [d for d in out
                if d.get("choices", [{}])[0].get("finish_reason") is not None]
    assert len(finishes) == 1
    assert finishes[0]["choices"][0]["finish_reason"] == "stop"
    # 不应注入超时通知（上游已正常 finish）
    assert not any("[DeepProxy]" in d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                   for d in out)
    assert res.timed_out is False


async def test_stream_with_idle_timeout_closes_upstream_on_early_close():
    """消费者提前 aclose（客户端断连）：上游异步生成器被 aclose，其 finally 确定性运行
    （释放连接），不依赖 GC。对齐 stream_one_turn 的同名契约。"""
    closed = {"hit": False}
    never = asyncio.Event()

    async def slow_gen():
        try:
            yield _delta_chunk(content="first")
            await never.wait()
            yield _delta_chunk(content="never")
        finally:
            closed["hit"] = True

    res = TurnResult()
    agen = stream_with_idle_timeout(
        slow_gen(), result=res,
        idle_timeout=30.0, first_chunk_timeout=30.0, heartbeat_seconds=30.0,
    )
    first = await agen.__anext__()
    assert first["choices"][0]["delta"] == {"content": "first"}
    await agen.aclose()
    assert closed["hit"] is True


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


async def test_continuation_resend_timeout_emits_notice_then_errored():
    """重发轮超时：先 yield 优雅超时通知（content + finish_reason=stop），再 yield
    STREAM_ERRORED 哨兵（供调用方标记不提交升格记账）。绝不静默返回空轮。"""
    from deep_proxy.cross_consult.client_stream import STREAM_ERRORED

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    cfg.cross_consult = CrossConsultConfig(
        enabled=True, pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    # CrossConsultConfig 无 validate_assignment：直接赋小数加速测试
    cfg.cross_consult.first_chunk_timeout_seconds = 0.2
    cfg.cross_consult.stream_heartbeat_seconds = 0.1
    cfg.providers["mimo"] = Provider(
        name="mimo", api_base="https://x", api_key="t", litellm_prefix="openai/",
        flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
    )
    source = cfg.providers["deepseek"]
    acc = StreamingReasoningAccumulator(request_messages=[])

    async def hanging_resend(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(1.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "use cc"}]}

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks",
               new=hanging_resend):
        frames = [f async for f in stream_cross_consult_continuation(
            initial_tool_calls=[_cc_tool_call()],
            body=body, source_provider=source, config=cfg,
            cc_config=cfg.cross_consult, accumulator=acc,
        )]

    # 优雅通知透传给客户端
    assert any("[DeepProxy]" in fr.get("choices", [{}])[0].get("delta", {}).get("content", "")
               for fr in frames)
    assert any(fr.get("choices", [{}])[0].get("finish_reason") == "stop" for fr in frames)
    # 末尾 STREAM_ERRORED 哨兵（按 identity 识别）
    assert any(fr is STREAM_ERRORED for fr in frames)


async def test_stream_one_turn_closes_upstream_on_early_close():
    """消费者提前关闭生成器（客户端断连 → GeneratorExit）时，上游异步生成器被
    aclose，其 finally 确定性运行（关闭 httpx 流 / 释放连接），不依赖 GC。"""
    closed = {"hit": False}
    # 用永不 set 的 Event 模拟"上游下一 chunk 迟迟不来"——cancel 时立即解除，
    # 不像 asyncio.sleep 那样在 loop 上残留计时器句柄（避免跨测试 loop 污染/flaky）。
    never = asyncio.Event()

    async def slow_gen():
        try:
            yield _delta_chunk(content="first")
            await never.wait()  # 永远阻塞，直到被 cancel
            yield _delta_chunk(content="never")
        finally:
            closed["hit"] = True

    res = TurnResult()
    agen = stream_one_turn(
        slow_gen(), res, tool_name="cross_consult",
        idle_timeout=30.0, first_chunk_timeout=30.0, heartbeat_seconds=30.0,
    )
    # 取首帧后立即关闭生成器（等价客户端断连）
    first = await agen.__anext__()
    assert first["choices"][0]["delta"] == {"content": "first"}
    await agen.aclose()
    assert closed["hit"] is True
