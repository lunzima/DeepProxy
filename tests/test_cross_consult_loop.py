"""Cross-Consult 响应路径拦截 + 重发循环测试（mock 流式 iter_litellm_chunks）。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def cfg_cross():
    from deep_proxy.config import ProxyConfig
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "deepseek/", "flash_model": "deepseek-v4-flash",
                         "pro_model": "deepseek-v4-pro"},
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
                     "pro_model": "mimo-v2.5-pro"},
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "y"},
        "cross_consult": {"enabled": True, "pairs": {"deepseek": "mimo", "mimo": "deepseek"}},
    })


def _tool_call_chunks(tool_call_id: str, args: dict):
    """伪造一段流式响应，吐一个 cross_consult tool_call。"""
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": tool_call_id, "type": "function",
            "function": {"name": "cross_consult", "arguments": json.dumps(args)},
        }]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]


def _text_chunks(text: str):
    """伪造一段流式响应，吐一段纯文本。"""
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]},
    ]


def _make_text_response(text: str):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def _make_tool_call_response(tool_call_id: str, args: dict):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "cross_consult",
                        "arguments": json.dumps(args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def _make_chunk_sequence_iter(*chunk_lists):
    """每次调用按顺序返回不同的 chunk 序列。供 iter_litellm_chunks mock。"""
    state = {"idx": 0}

    def factory(config, body, *, provider=None):
        i = state["idx"]
        state["idx"] += 1
        cl = chunk_lists[i] if i < len(chunk_lists) else chunk_lists[-1]

        async def gen():
            for c in cl:
                yield c
        return gen()
    return factory


async def test_chat_completions_executes_cross_consult_and_resends(cfg_cross):
    """主 LLM 调 cross_consult → DeepProxy 执行 consult → 重发原 provider → 拿到最终文本。

    路径覆盖：
      - 主 provider 的第 1 次调用（chat_completions 初始）走非流式 call_litellm
      - executor 走 streaming.iter_litellm_chunks
      - 重发走 stream_aggregated_call → streaming.iter_litellm_chunks
    """
    from deep_proxy.router import DeepProxyRouter

    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    # 主 provider 初始（非流式）
    initial_response = _make_tool_call_response("tc1", {"question": "what is X?"})

    # 两次流式调用：
    #   1) executor 调对偶 provider → 返回 "external answer"
    #   2) 重发原 provider → 返回最终文本
    stream_iter = _make_chunk_sequence_iter(
        _text_chunks("external answer"),
        _text_chunks("final answer using external answer"),
    )

    # cc 活跃时初始调用走 aggregate_stream_to_response（流式聚合）；重发走 iter_litellm_chunks
    with patch("deep_proxy.router.aggregate_stream_to_response",
               new=AsyncMock(return_value=initial_response)), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=stream_iter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Use cross_consult to learn X."}],
        }
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "final answer using external answer"


async def test_chat_completions_passes_through_when_no_cross_consult_call(cfg_cross):
    """普通响应不含 cross_consult tool_call 时，行为与之前一致。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    with patch("deep_proxy.router.aggregate_stream_to_response",
               new=AsyncMock(return_value=_make_text_response("hi"))):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "say hi"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "hi"


async def test_chat_completions_handles_consult_error_as_tool_result(cfg_cross):
    """consult 失败时错误字符串作为 tool_result 注入；主 provider 仍能继续。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    initial_response = _make_tool_call_response("tc1", {"question": "what?"})

    # 主 provider 重发：流式返回最终文本
    resend_iter = _make_chunk_sequence_iter(_text_chunks("final after error"))

    async def fake_executor(*args, **kwargs):
        return "[DeepProxy cross_consult error] upstream failed: simulated"

    with patch("deep_proxy.router.aggregate_stream_to_response",
               new=AsyncMock(return_value=initial_response)), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult",
               new=AsyncMock(side_effect=fake_executor)), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=resend_iter):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "ask"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "final after error"


async def test_iter_chat_chunks_streams_cross_consult_live(cfg_cross):
    """cc 激活时，初始 content/reasoning + 重发 content 逐帧到达客户端，cc 工具帧不可见。"""
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter

    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    async def initial_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"reasoning_content": "想一下"},
                            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"content": "让我咨询"},
                            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "cc1", "type": "function",
             "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}]},
            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

    async def resend_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "综合答案"},
                            "finish_reason": "stop"}]}

    calls = {"n": 0}

    def dispatch(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        return initial_stream(config, body) if calls["n"] == 1 else resend_stream(config, body)

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.router.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "use cc"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]

    deltas = [fr.get("choices", [{}])[0].get("delta", {}) for fr in frames if "choices" in fr]
    assert {"reasoning_content": "想一下"} in deltas
    assert {"content": "让我咨询"} in deltas
    assert {"content": "综合答案"} in deltas
    assert not any("tool_calls" in d and any(
        (tc.get("function") or {}).get("name") == "cross_consult" for tc in d["tool_calls"]
    ) for d in deltas)


async def test_iter_chat_chunks_no_cc_call_passes_through(cfg_cross):
    """初始流不含 cc 调用：content 透传 + 终轮 finish_reason，行为等价直通。"""
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    async def plain(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "你好"},
                            "finish_reason": "stop"}]}

    with patch("deep_proxy.router.iter_litellm_chunks", new=plain):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hi"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]
    assert any(fr.get("choices", [{}])[0].get("delta", {}).get("content") == "你好"
               for fr in frames)
    assert any(fr.get("choices", [{}])[0].get("finish_reason") == "stop" for fr in frames)


async def test_iter_chat_chunks_heartbeat_during_consult(cfg_cross):
    """consult 执行慢时，客户端收到心跳帧。"""
    import asyncio
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    router.config.streaming.heartbeat_seconds = 1
    provider = cfg_cross.providers["deepseek"]

    async def initial_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "cc1", "type": "function",
             "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}]},
            "finish_reason": "tool_calls"}]}

    async def resend_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}

    calls = {"n": 0}
    def dispatch(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        return initial_stream(config, body) if calls["n"] == 1 else resend_stream(config, body)

    async def slow_consult(**kw):
        await asyncio.sleep(1.5)
        return "外部视角"

    with patch("deep_proxy.router.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=slow_consult):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "use cc"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]
    assert any(f == {"_dp_heartbeat": True} for f in frames)


async def test_iter_chat_chunks_timeout_does_not_commit_upgrade(cfg_cross):
    """I-1 回归：初始轮首 chunk 超时（result.errored）不是干净完成，
    不得提交升格记账（_commit_pending_upgrade 不被调用）。"""
    import asyncio
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    # 心跳 1s < 首 chunk 预算 2s → 先发一个心跳，再在第二个 tick 超时
    router.config.streaming.first_chunk_timeout_seconds = 2
    router.config.streaming.max_retries = 0   # 首 chunk 超时立即硬错误，不重发
    router.config.streaming.heartbeat_seconds = 1
    provider = cfg_cross.providers["deepseek"]

    async def never_first(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(5)  # 首 chunk 永不在预算内到达
        yield {"choices": [{"index": 0, "delta": {"content": "late"},
                            "finish_reason": "stop"}]}

    committed = {"hit": False}
    def fake_commit(b):
        committed["hit"] = True

    with patch("deep_proxy.router.iter_litellm_chunks", new=never_first), \
         patch.object(router, "_commit_pending_upgrade", new=fake_commit):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "use cc"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]

    # 超时前发过心跳，但最终超时 → 不提交升格记账（I-1 核心断言）
    assert any(f == {"_dp_heartbeat": True} for f in frames)
    assert committed["hit"] is False


async def test_resend_loop_uses_streaming_iter(cfg_cross):
    """关键回归：重发循环必须经过 streaming.iter_litellm_chunks（避免墙钟超时
    在深度思考期间错误地杀掉响应）。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    initial_response = _make_tool_call_response("tc1", {"question": "q"})

    iter_calls = {"count": 0}

    def factory(config, body, *, provider=None):
        iter_calls["count"] += 1

        async def gen():
            for c in _text_chunks("done"):
                yield c
        return gen()

    async def fake_executor(*args, **kwargs):
        return "external"

    # 初始调用走 aggregate_stream_to_response（mock 返回带 tool_call 的初始响应）；
    # 重发走 streaming.iter_litellm_chunks
    with patch("deep_proxy.router.aggregate_stream_to_response",
               new=AsyncMock(return_value=initial_response)), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult",
               new=AsyncMock(side_effect=fake_executor)), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=factory):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "ask"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        await router.chat_completions(body, provider=provider)

    # executor 被 mock 掉了不走 streaming；resend 必须经过一次 streaming iter
    assert iter_calls["count"] == 1, (
        f"expected exactly 1 streaming resend, got {iter_calls['count']}"
    )


async def test_loop_calls_process_response_on_each_iteration(cfg_cross):
    """C3 regression: 每次重发响应都应过 process_response 再 append 到 history。

    注：此测试直接调 execute_cross_consult_loop 并传入 call_litellm_fn=fake_call，
    验证 loop 主循环逻辑——不经过 router.py 的流式封装路径。
    """
    from deep_proxy.cross_consult.interceptor import execute_cross_consult_loop

    process_calls = []

    def fake_process_response(resp, *, provider=None):
        process_calls.append(resp)
        return resp

    main_responses = [
        _make_tool_call_response("tc2", {"question": "q2"}),
        _make_text_response("done"),
    ]

    async def fake_call(config, body, *, provider=None):
        return main_responses.pop(0)

    async def fake_consult(*args, **kwargs):
        return "external"

    initial = _make_tool_call_response("tc1", {"question": "q1"})

    with patch("deep_proxy.cross_consult.interceptor.execute_consult",
               side_effect=fake_consult):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "go"}],
                "stream": False}
        await execute_cross_consult_loop(
            body=body,
            initial_response=initial,
            source_provider=cfg_cross.providers["deepseek"],
            config=cfg_cross,
            cc_config=cfg_cross.cross_consult,
            call_litellm_fn=fake_call,
            process_response_fn=fake_process_response,
        )

    assert len(process_calls) == 2, (
        f"expected 2 process_response invocations, got {len(process_calls)}"
    )


async def test_loop_drops_non_cc_tool_calls_from_resend_history(cfg_cross):
    """同轮混用真实工具 + cross_consult：resend 历史的 assistant 消息只保留 cc 调用，
    避免真实工具 tool_call 无 tool_result 悬空 → 上游 400（审核 #11）。"""
    from deep_proxy.cross_consult.interceptor import execute_cross_consult_loop

    initial = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "real1", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "cc1", "type": "function",
         "function": {"name": "cross_consult", "arguments": json.dumps({"question": "q"})}},
    ]}, "finish_reason": "tool_calls"}]}

    async def fake_call(config, body, *, provider=None):
        return _make_text_response("done")

    async def fake_consult(*a, **k):
        return "external"

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "go"}], "stream": False}
    with patch("deep_proxy.cross_consult.interceptor.execute_consult",
               side_effect=fake_consult):
        await execute_cross_consult_loop(
            body=body, initial_response=initial,
            source_provider=cfg_cross.providers["deepseek"], config=cfg_cross,
            cc_config=cfg_cross.cross_consult, call_litellm_fn=fake_call,
        )

    asst = [m for m in body["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst
    tc_ids = {tc["id"] for m in asst for tc in m["tool_calls"]}
    tool_result_ids = {m["tool_call_id"] for m in body["messages"]
                       if m.get("role") == "tool"}
    assert tc_ids <= tool_result_ids, f"悬空 tool_call: {tc_ids - tool_result_ids}"
    assert "real1" not in tc_ids   # 真实工具调用不进 resend 历史
    assert "cc1" in tc_ids


def test_drop_cc_tool_calls_helper():
    from deep_proxy.cross_consult.interceptor import drop_cc_tool_calls
    tcs = [
        {"id": "r", "function": {"name": "read_file"}},
        {"id": "c", "function": {"name": "cross_consult"}},
    ]
    assert [tc["id"] for tc in drop_cc_tool_calls(tcs, "cross_consult")] == ["r"]
    assert drop_cc_tool_calls(None, "cross_consult") == []


async def test_loop_hard_limit_strips_unresolved_cc_tool_call(cfg_cross):
    """硬轮次上限退出：返回的响应不得残留未执行的 cross_consult tool_call
    （客户端无法执行虚拟工具）（审核：hard-limit cc-call leak）。"""
    from deep_proxy.cross_consult.interceptor import execute_cross_consult_loop

    async def fake_call(config, body, *, provider=None):
        return _make_tool_call_response("ccN", {"question": "q"})  # 每轮都发 cc 调用

    async def fake_consult(*a, **k):
        return "external"

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "go"}], "stream": False}
    with patch("deep_proxy.cross_consult.interceptor.execute_consult",
               side_effect=fake_consult):
        result = await execute_cross_consult_loop(
            body=body, initial_response=_make_tool_call_response("cc0", {"question": "q"}),
            source_provider=cfg_cross.providers["deepseek"], config=cfg_cross,
            cc_config=cfg_cross.cross_consult, call_litellm_fn=fake_call)

    msg = result["choices"][0]["message"]
    names = [(tc.get("function") or {}).get("name") for tc in (msg.get("tool_calls") or [])]
    assert "cross_consult" not in names


async def test_streaming_final_chunk_includes_reasoning_content_when_present(cfg_cross):
    """I4 regression: 真流式 cross_consult 路径应将 reasoning_content 帧逐帧透传到客户端。"""
    from deep_proxy.router import DeepProxyRouter

    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    # 初始流：吐 cross_consult tool_call
    initial_stream_chunks = _tool_call_chunks("tc1", {"question": "q"})

    async def initial_iter(*args, **kwargs):
        for c in initial_stream_chunks:
            yield c

    # 重发流：吐 content + reasoning_content
    resend_chunks = [
        {"choices": [{"index": 0,
                      "delta": {"reasoning_content": "step by step thinking"},
                      "finish_reason": None}]},
        {"choices": [{"index": 0,
                      "delta": {"content": "final content"},
                      "finish_reason": "stop"}]},
    ]

    async def consult_ok(**kw):
        return "ext"

    # client_stream.iter_litellm_chunks is called only once: for the resend after consult.
    def dispatch(config, body, *, _accumulator=None, provider=None):
        async def gen():
            for c in resend_chunks:
                yield c
        return gen()

    with patch("deep_proxy.router.iter_litellm_chunks", new=initial_iter), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        out_chunks = []
        async for chunk in router.iter_chat_chunks(body, provider=provider):
            out_chunks.append(chunk)

    reasoning_emitted = False
    for c in out_chunks:
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("reasoning_content") == "step by step thinking":
                reasoning_emitted = True
    assert reasoning_emitted, "streaming real-time forwarding dropped reasoning_content"


async def test_chat_completions_stream_serializes_heartbeat_as_sse_comment():
    """心跳 sentinel -> SSE 注释帧（: keep-alive），不是 data: 帧。"""
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    from deep_proxy.config import ProxyConfig, normalize_legacy_config

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    router = DeepProxyRouter(cfg)

    async def fake_iter(body, *, provider=None):
        yield {"_dp_heartbeat": True}
        yield {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}

    with patch.object(router, "iter_chat_chunks", new=fake_iter):
        out = [s async for s in router.chat_completions_stream({}, provider=None)]

    assert ": keep-alive\n\n" in out
    assert not any(s.startswith("data: ") and "_dp_heartbeat" in s for s in out)
    assert any(s.startswith("data: ") and '"content": "hi"' in s for s in out)
    assert out[-1] == "data: [DONE]\n\n"
