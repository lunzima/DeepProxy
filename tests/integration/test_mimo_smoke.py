"""MiMo 真实 API 烟测（需 MIMO_API_KEY 环境变量）。

跑法：
    MIMO_API_KEY=tp-... pytest tests/integration/test_mimo_smoke.py -v

默认 pytest 排除 tests/integration/（见 pytest.ini）。
"""
from __future__ import annotations

import os

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.providers import Provider
from deep_proxy.router import DeepProxyRouter

MIMO_KEY = os.getenv("MIMO_API_KEY")
pytestmark = pytest.mark.skipif(
    not MIMO_KEY,
    reason="需 MIMO_API_KEY 环境变量",
)


@pytest.fixture
def cfg_with_mimo() -> ProxyConfig:
    return ProxyConfig.model_validate({
        "providers": {
            "mimo": {
                "name": "mimo",
                "api_base": "https://token-plan-cn.xiaomimimo.com/v1",
                "api_key": MIMO_KEY,
                "litellm_prefix": "openai/",
                "flash_model": "mimo-v2.5",
                "pro_model": "mimo-v2.5-pro",
                "reasoning_effort_field": "reasoning_effort",
                "reasoning_effort_value": "high",
                "max_output_tokens": 128000,
                "allowed_extra_params": ["reasoning_effort", "thinking"],
            },
        },
        "ports": [
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "sk-not-used-in-mimo-smoke"},
    })


async def test_mimo_non_stream_chat(cfg_with_mimo):
    router = DeepProxyRouter(cfg_with_mimo)
    provider = cfg_with_mimo.providers["mimo"]
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "Say 'hi' once."}],
        "max_tokens": 50,
    }
    body = await router.prepare_request(
        body, sampling_profile=cfg_with_mimo.creative_sampling, provider=provider,
    )
    result = await router.chat_completions(body, provider=provider)
    assert result["choices"][0]["message"]["content"]
    await router.close()


async def test_mimo_stream_chat(cfg_with_mimo):
    router = DeepProxyRouter(cfg_with_mimo)
    provider = cfg_with_mimo.providers["mimo"]
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "Count 1 to 3."}],
        "max_tokens": 30,
        "stream": True,
    }
    body = await router.prepare_request(
        body, sampling_profile=cfg_with_mimo.creative_sampling, provider=provider,
    )
    chunks = []
    async for chunk in router.iter_chat_chunks(body, provider=provider):
        chunks.append(chunk)
    assert len(chunks) > 0
    # 至少有一个 chunk 含 content delta
    assert any(
        (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
        for chunk in chunks
    )
    await router.close()


async def test_mimo_with_tools(cfg_with_mimo):
    router = DeepProxyRouter(cfg_with_mimo)
    provider = cfg_with_mimo.providers["mimo"]
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "What's the weather in Beijing?"}],
        "max_tokens": 100,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
    }
    body = await router.prepare_request(
        body, sampling_profile=cfg_with_mimo.creative_sampling, provider=provider,
    )
    result = await router.chat_completions(body, provider=provider)
    msg = result["choices"][0]["message"]
    assert msg.get("tool_calls") or msg.get("content")
    await router.close()
