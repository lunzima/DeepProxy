"""Cross-Consult 响应路径拦截 + 重发循环测试（mock LiteLLM）。"""
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


def _make_text_response(text: str):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


async def test_chat_completions_executes_cross_consult_and_resends(cfg_cross):
    """主 LLM 调 cross_consult → DeepProxy 执行 consult → 重发原 provider → 拿到最终文本。"""
    from deep_proxy.router import DeepProxyRouter

    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    # 模拟三次 call_litellm：
    # 1) 主 provider 返回 cross_consult tool_call
    # 2) executor 调对偶 provider 返回 "external answer"
    # 3) 主 provider 重发后返回最终文本
    responses = [
        _make_tool_call_response("tc1", {"question": "what is X?"}),
        _make_text_response("external answer"),
        _make_text_response("final answer using external answer"),
    ]

    async def fake_call_litellm(config, body, *, provider=None):
        return responses.pop(0)

    with patch("deep_proxy.router.call_litellm", new=AsyncMock(side_effect=fake_call_litellm)), \
         patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(side_effect=fake_call_litellm)):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Use cross_consult to learn X."}],
        }
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "final answer using external answer"
    # responses 应该全部消耗完
    assert len(responses) == 0


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

    responses_main = [
        _make_tool_call_response("tc1", {"question": "what?"}),
        _make_text_response("final after error"),
    ]

    async def fake_main(config, body, *, provider=None):
        return responses_main.pop(0)

    async def fake_executor(*args, **kwargs):
        return "[DeepProxy cross_consult error] upstream failed: simulated"

    with patch("deep_proxy.router.call_litellm", new=AsyncMock(side_effect=fake_main)), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult",
               new=AsyncMock(side_effect=fake_executor)):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "ask"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "final after error"
    assert len(responses_main) == 0
