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

    with patch("deep_proxy.router.call_litellm",
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

    with patch("deep_proxy.router.call_litellm",
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

    with patch("deep_proxy.router.call_litellm",
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


async def test_iter_chat_chunks_intercepts_cross_consult_in_stream(cfg_cross):
    """流式响应含 cross_consult tool_call 时，DeepProxy 走"内部流式聚合"补完模式。

    路径覆盖：
      - 主 provider 初始流式：含 cross_consult tool_call
      - executor 流式调对偶 provider → 返回 "external"
      - 重发流式调原 provider → 返回最终文本 "final"
    """
    from deep_proxy.router import DeepProxyRouter

    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    initial_stream_chunks = _tool_call_chunks("tc1", {"question": "what?"})

    async def initial_iter(*args, **kwargs):
        for c in initial_stream_chunks:
            yield c

    # executor + resend 共享同一组流式 mock（按调用次数返回不同内容）
    cc_stream_iter = _make_chunk_sequence_iter(
        _text_chunks("external"),
        _text_chunks("final"),
    )

    with patch("deep_proxy.router.iter_litellm_chunks", new=initial_iter), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=cc_stream_iter):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        out_chunks = []
        async for chunk in router.iter_chat_chunks(body, provider=provider):
            out_chunks.append(chunk)

    contents = []
    for c in out_chunks:
        for ch in c.get("choices") or []:
            d = ch.get("delta") or ch.get("message") or {}
            v = d.get("content")
            if v:
                contents.append(v)
    assert "final" in "".join(contents)


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

    with patch("deep_proxy.router.call_litellm",
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


async def test_streaming_final_chunk_includes_reasoning_content_when_present(cfg_cross):
    """I4 regression: 流式 cross_consult 合成 chunk 应保留 reasoning_content。"""
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
    # executor 流：随便给一段
    executor_chunks = _text_chunks("ext")

    cc_stream_iter = _make_chunk_sequence_iter(executor_chunks, resend_chunks)

    with patch("deep_proxy.router.iter_litellm_chunks", new=initial_iter), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=cc_stream_iter):
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
    assert reasoning_emitted, "streaming final chunk dropped reasoning_content"
