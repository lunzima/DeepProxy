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


def _text_chunks(text: str):
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"},
                      "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": text},
                      "finish_reason": "stop"}]},
    ]


def _stream_router_by_provider(captured: dict, target_name: str):
    """流式 mock 工厂：consult（provider.name == target_name）记录 body；
    其它（resend，provider == source）正常返回 final 文本流。"""
    def factory(config, body, *, provider=None):
        if provider is not None and provider.name == target_name:
            captured["model"] = body["model"]
            captured["provider_name"] = provider.name

            async def gen():
                for c in _text_chunks("external answer"):
                    yield c
            return gen()

        async def gen():
            for c in _text_chunks("final"):
                yield c
        return gen()
    return factory


async def test_deepseek_source_consults_mimo_pro(cfg_sym):
    """source=deepseek → target=mimo-v2.5-pro。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_sym)
    source = cfg_sym.providers["deepseek"]

    captured_consult_body: dict = {}

    # 主 provider 初始（非流式）返回 tool_call
    initial = _tc("t", {"question": "q"})

    with patch("deep_proxy.router.call_litellm",
               new=AsyncMock(return_value=initial)), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=_stream_router_by_provider(captured_consult_body, "mimo")):
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

    captured_consult_body: dict = {}

    initial = _tc("t", {"question": "q"})

    with patch("deep_proxy.router.call_litellm",
               new=AsyncMock(return_value=initial)), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=_stream_router_by_provider(captured_consult_body, "deepseek")):
        body = {"model": "mimo-v2.5",
                "messages": [{"role": "user", "content": "go"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_sym.creative_sampling, provider=source,
        )
        await router.chat_completions(body, provider=source)

    assert captured_consult_body["model"] == "deepseek-v4-pro"
    assert captured_consult_body["provider_name"] == "deepseek"
