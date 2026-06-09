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
    stream_one_turn, TurnResult, stream_with_idle_timeout,
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


from deep_proxy.cross_consult.client_stream import (  # noqa: E402
    consume_with_heartbeat, _Timeout,
)


async def test_consume_with_heartbeat_raises_idle_on_reasoning():
    """引擎内部自适应：见 reasoning_content 后 idle 升到 reasoning_idle；
    推理停顿 > content idle 但 ≤ reasoning_idle 不超时。"""
    async def gen():
        yield _delta_chunk(reasoning_content="想")
        await asyncio.sleep(0.3)          # > idle 0.1，< reasoning_idle 2.0
        yield _delta_chunk(content="答")

    items = [it async for it in consume_with_heartbeat(
        gen(), idle_timeout=0.1, reasoning_idle=2.0,
        first_chunk_timeout=5.0, heartbeat_seconds=0.05, log_label="t",
    )]
    assert not any(isinstance(it, _Timeout) for it in items)
    chunks = [it for it in items if isinstance(it, dict) and "choices" in it]
    assert any(c["choices"][0]["delta"].get("content") == "答" for c in chunks)


async def test_stream_one_turn_reasoning_idle_tolerates_gap():
    """显式 reasoning_idle：检测到 reasoning 后，超过 content idle 但 ≤ reasoning_idle
    的停顿不触发超时。"""
    async def reasoning_then_pause():
        yield _delta_chunk(reasoning_content="思考")
        await asyncio.sleep(0.35)          # > idle 0.15，< reasoning_idle 2.0
        yield _delta_chunk(content="答案")
        yield _finish_chunk("stop")

    res = TurnResult()
    [f async for f in stream_one_turn(  # 驱动流，断言走 res
        reasoning_then_pause(), res, tool_name="cross_consult",
        idle_timeout=0.15, reasoning_idle=2.0,
        first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is False
    assert res.content == "答案"


async def test_stream_one_turn_no_timeout_after_upstream_finish():
    """上游发完 finish_reason 后又挂连接（finish-then-hang）：本轮逻辑已正常结束，
    不得标 timed_out（否则 stream_turn_with_retry 会把成功轮误报成硬错误/504）。
    对齐 stream_with_idle_timeout 的同名守卫。"""
    async def finish_then_hang():
        yield _delta_chunk(content="答案")
        yield _finish_chunk("stop")
        await asyncio.Event().wait()  # 发完 finish 后永久挂起（不抛 StopAsyncIteration）

    res = TurnResult()
    [f async for f in stream_one_turn(
        finish_then_hang(), res, tool_name="cross_consult",
        idle_timeout=0.2, first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is False
    assert res.errored is False
    assert res.finish_reason == "stop"
    assert res.content == "答案"


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


async def test_stream_with_idle_timeout_detection_only_no_notice():
    """detection-only：超时只写 result 元数据 + 发心跳，不再注入 [DeepProxy] 通知 /
    clean finish（旧"请重试"通知对 agent 结构上不可触发重试，已废弃）。重试/硬错误策略
    交给 stream_with_retry。"""
    async def hang_gen():
        await asyncio.sleep(10.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    out = [f async for f in stream_with_idle_timeout(
        hang_gen(), result=res,
        idle_timeout=5.0, first_chunk_timeout=0.2, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is True
    assert res.timeout_phase == "first_chunk"
    assert not any("[DeepProxy]" in (d.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
                   for d in out if "choices" in d)
    assert not any(d.get("choices", [{}])[0].get("finish_reason") == "stop"
                   for d in out if "choices" in d)
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


async def test_continuation_resend_timeout_hard_errors():
    """重发轮 pre-content 持续挂死、总预算耗尽 → 发**硬错误帧**（{"error":...} 透传给
    客户端使 SDK 抛错），不再注入已废弃的优雅通知 / clean stop / STREAM_ERRORED。"""
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    cfg.cross_consult = CrossConsultConfig(
        enabled=True, pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    # 超时取自 StreamingConfig（合并后单一配置）；直接赋小数加速测试
    cfg.streaming.first_chunk_timeout_seconds = 0.2
    cfg.streaming.heartbeat_seconds = 0.1
    cfg.streaming.max_retries = 1   # 1 次重发后硬错误
    cfg.providers["mimo"] = Provider(
        name="mimo", api_base="https://x", api_key="t", litellm_prefix="openai/",
        flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
    )
    source = cfg.providers["deepseek"]
    acc = StreamingReasoningAccumulator(request_messages=[])

    async def hanging_resend(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(5.0)
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

    # 硬错误帧透传给客户端（is_error_frame: error 是 dict 且无 choices）
    assert any(isinstance(fr.get("error"), dict) and not fr.get("choices") for fr in frames)
    # 不再注入旧的优雅通知 / clean stop
    assert not any("[DeepProxy]" in (fr.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
                   for fr in frames if "choices" in fr)
    assert not any(fr.get("choices", [{}])[0].get("finish_reason") == "stop"
                   for fr in frames if "choices" in fr)


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


def _cc_continuation_cfg():
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
    return cfg


async def test_continuation_appended_assistant_carries_reasoning_content():
    """回归：cc 重发轮追加的 assistant 消息（携带 cc tool_call）必须带**非空**
    reasoning_content。DeepSeek thinking 模式要求每条历史 assistant 回传 reasoning_content，
    缺失 → 400 'The reasoning_content in the thinking mode must be passed back to the API.'

    重发路径直接 iter_litellm_chunks、绕过 prepare_request/ensure_reasoning_content_persistence，
    故须在 continuation 内自补（真实思考缺失时用 dummy 兜底）。"""
    cfg = _cc_continuation_cfg()
    source = cfg.providers["deepseek"]          # has_reasoning_content=True
    acc = StreamingReasoningAccumulator(request_messages=[])

    async def resend_iter(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "答案"},
                            "finish_reason": "stop"}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "use cc"}]}

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks",
               new=resend_iter):
        _ = [f async for f in stream_cross_consult_continuation(
            initial_tool_calls=[_cc_tool_call()],
            body=body, source_provider=source, config=cfg,
            cc_config=cfg.cross_consult, accumulator=acc,
        )]

    from deep_proxy.compatibility.reasoning_handler import _DUMMY_REASONING

    assistant_cc_msgs = [
        m for m in body["messages"]
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert assistant_cc_msgs, "应追加携带 cc tool_call 的 assistant 消息"
    # 兜底走既有 ensure_reasoning_content_persistence 设施（而非内联自造），故为其规范 dummy
    assert all(m.get("reasoning_content") == _DUMMY_REASONING for m in assistant_cc_msgs), \
        "缺真实思考时须由 ensure_reasoning_content_persistence 注入规范 dummy"


async def test_continuation_preserves_real_reasoning_content():
    """有真实思考时 continuation 须保留之（而非 dummy），与非流式
    aggregate_stream_to_response 携带 reasoning_content 的行为对齐。"""
    cfg = _cc_continuation_cfg()
    source = cfg.providers["deepseek"]
    acc = StreamingReasoningAccumulator(request_messages=[])

    async def resend_iter(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "答案"},
                            "finish_reason": "stop"}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "use cc"}]}

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks",
               new=resend_iter):
        _ = [f async for f in stream_cross_consult_continuation(
            initial_tool_calls=[_cc_tool_call()],
            initial_reasoning="模型本轮的真实思考",
            body=body, source_provider=source, config=cfg,
            cc_config=cfg.cross_consult, accumulator=acc,
        )]

    first_cc = next(
        m for m in body["messages"]
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert first_cc["reasoning_content"] == "模型本轮的真实思考"
