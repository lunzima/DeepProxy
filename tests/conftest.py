"""共享 pytest fixtures。"""
from __future__ import annotations

import pytest

from deep_proxy.config import DeepSeekConfig, ProxyConfig
from deep_proxy.router import DeepProxyRouter


@pytest.fixture
def cfg() -> ProxyConfig:
    return ProxyConfig(
        deepseek=DeepSeekConfig(api_key="sk-test", api_base="https://api.deepseek.com")
    )


@pytest.fixture
def router(cfg: ProxyConfig) -> DeepProxyRouter:
    return DeepProxyRouter(cfg)
