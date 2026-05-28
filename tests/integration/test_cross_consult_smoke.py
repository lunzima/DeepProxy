"""Cross-Consult 真实双 provider 集成 smoke。

需 DEEPSEEK_API_KEY + MIMO_API_KEY 同时设置。
"""
from __future__ import annotations

import os

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.router import DeepProxyRouter

DS_KEY = os.getenv("DEEPSEEK_API_KEY")
MIMO_KEY = os.getenv("MIMO_API_KEY")
pytestmark = pytest.mark.skipif(
    not (DS_KEY and MIMO_KEY),
    reason="需 DEEPSEEK_API_KEY 与 MIMO_API_KEY 同时设置",
)


@pytest.fixture
def cfg_dual_real():
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek",
                "api_base": "https://api.deepseek.com",
                "api_key": DS_KEY,
                "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash",
                "pro_model": "deepseek-v4-pro",
                "legacy_aliases": {
                    "deepseek-chat": {"thinking": {"type": "disabled"}},
                    "deepseek-reasoner": {"thinking": {"type": "enabled"}},
                },
            },
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
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": DS_KEY},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
            "max_calls_per_request": 2,
        },
    })


async def test_cross_consult_real_roundtrip_deepseek_to_mimo(cfg_dual_real):
    """source=deepseek，提示词鼓励调用 cross_consult，验证 round-trip。

    此测试依赖 LLM 决策——会自然失败的情况：deepseek 不调 cross_consult。
    若实际 round-trip 未触发，测试仍通过（只验证 happy path 无崩溃）。
    """
    router = DeepProxyRouter(cfg_dual_real)
    provider = cfg_dual_real.providers["deepseek"]
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "user",
            "content": (
                "你被允许调用 cross_consult 向异家族模型请求第二视角。"
                "请用 cross_consult 询问：'用一句中文回答：什么是熵？'，"
                "然后把对方答案原样转述。"
            ),
        }],
        "max_tokens": 600,
    }
    body = await router.prepare_request(
        body, sampling_profile=cfg_dual_real.precise_sampling, provider=provider,
    )
    result = await router.chat_completions(body, provider=provider)
    msg = result["choices"][0]["message"]
    # 主响应应非空（不强制要求触发 cross_consult，但流程必须无崩溃）
    assert msg.get("content") or msg.get("tool_calls") or msg.get("reasoning_content")
    await router.close()
