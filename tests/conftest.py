"""共享 pytest fixtures。"""
from __future__ import annotations

import pytest

from deep_proxy.config import DeepSeekConfig, ProxyConfig, normalize_legacy_config
from deep_proxy.providers import Provider
from deep_proxy.router import DeepProxyRouter


@pytest.fixture
def cfg() -> ProxyConfig:
    """单 provider 老格式 cfg（向后兼容测试用）。"""
    raw = normalize_legacy_config({
        "deepseek": {"api_key": "sk-test", "api_base": "https://api.deepseek.com"},
    })
    return ProxyConfig.model_validate(raw)


@pytest.fixture
def router(cfg: ProxyConfig) -> DeepProxyRouter:
    return DeepProxyRouter(cfg)


@pytest.fixture
def cfg_dual() -> ProxyConfig:
    """双 provider cfg（deepseek + mimo）测试用。"""
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek",
                "api_base": "https://api.deepseek.com",
                "api_key": "sk-test",
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
                "api_key": "tp-test",
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
        "deepseek": {"api_key": "sk-test"},
    })


@pytest.fixture
def router_dual(cfg_dual: ProxyConfig) -> DeepProxyRouter:
    return DeepProxyRouter(cfg_dual)


@pytest.fixture
def provider_deepseek() -> Provider:
    return Provider(
        name="deepseek",
        api_base="https://api.deepseek.com",
        api_key="sk-test",
        litellm_prefix="deepseek/",
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
    )


@pytest.fixture
def provider_mimo() -> Provider:
    return Provider(
        name="mimo",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        api_key="tp-test",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
        reasoning_effort_field="reasoning_effort",
        reasoning_effort_value="high",
        max_output_tokens=128000,
        allowed_extra_params=["reasoning_effort", "thinking"],
    )
