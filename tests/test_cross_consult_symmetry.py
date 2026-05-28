"""Cross-Consult 双向对称性：deepseek→mimo 与 mimo→deepseek 应行为对称。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def cfg_sym():
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


def _tc(tcid, args):
    return {"choices": [{"message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": tcid, "type": "function",
                        "function": {"name": "cross_consult",
                                     "arguments": json.dumps(args)}}]},
        "finish_reason": "tool_calls"}]}


def _text(t):
    return {"choices": [{"message": {"role": "assistant", "content": t},
                          "finish_reason": "stop"}]}


async def test_deepseek_source_consults_mimo_pro(cfg_sym):
    """source=deepseek → target=mimo-v2.5-pro。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_sym)
    source = cfg_sym.providers["deepseek"]

    captured_consult_body = {}

    async def fake_main(config, body, *, provider=None):
        # If conversation has a tool result, return final; else return tool call
        if any(m.get("role") == "tool" for m in body["messages"]):
            return _text("final")
        return _tc("t", {"question": "q"})

    async def fake_executor_call(config, body, *, provider=None):
        captured_consult_body["model"] = body["model"]
        captured_consult_body["provider_name"] = provider.name
        return _text("external answer")

    with patch("deep_proxy.router.call_litellm",
               new=AsyncMock(side_effect=fake_main)), \
         patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(side_effect=fake_executor_call)):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "go"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_sym.precise_sampling, provider=source,
        )
        await router.chat_completions(body, provider=source)

    assert captured_consult_body["model"] == "mimo-v2.5-pro"
    assert captured_consult_body["provider_name"] == "mimo"


async def test_mimo_source_consults_deepseek_pro(cfg_sym):
    """source=mimo → target=deepseek-v4-pro。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_sym)
    source = cfg_sym.providers["mimo"]

    captured_consult_body = {}

    async def fake_main(config, body, *, provider=None):
        if any(m.get("role") == "tool" for m in body["messages"]):
            return _text("final")
        return _tc("t", {"question": "q"})

    async def fake_executor_call(config, body, *, provider=None):
        captured_consult_body["model"] = body["model"]
        captured_consult_body["provider_name"] = provider.name
        return _text("external answer")

    with patch("deep_proxy.router.call_litellm",
               new=AsyncMock(side_effect=fake_main)), \
         patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(side_effect=fake_executor_call)):
        body = {"model": "mimo-v2.5",
                "messages": [{"role": "user", "content": "go"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_sym.creative_sampling, provider=source,
        )
        await router.chat_completions(body, provider=source)

    assert captured_consult_body["model"] == "deepseek-v4-pro"
    assert captured_consult_body["provider_name"] == "deepseek"
