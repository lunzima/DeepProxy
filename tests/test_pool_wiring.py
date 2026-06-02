"""main.py 层的 pool 选择 + port 接线（_binding_for_request）。"""
from __future__ import annotations

import types

import deep_proxy.main as main
from deep_proxy.config import ProxyConfig


def _fake_request(port: int):
    return types.SimpleNamespace(scope={"server": ("0.0.0.0", port)})


def _pooled_cfg() -> ProxyConfig:
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek", "api_base": "https://api.deepseek.com",
                "api_key": "sk-test", "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash", "pro_model": "deepseek-v4-pro",
            },
            "mimo": {
                "name": "mimo", "api_base": "https://m/v1", "api_key": "tp-test",
                "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
                "pro_model": "mimo-v2.5-pro",
            },
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative", "model_pool": [
                {"provider": "deepseek", "model": "deepseek-v4-flash", "weight": 1},
                {"provider": "mimo", "model": "mimo-v2.5", "weight": 1},
                {"provider": "deepseek", "model": "deepseek-v4-pro", "weight": 1},
                {"provider": "mimo", "model": "mimo-v2.5-pro", "weight": 1},
            ]},
        ],
        "deepseek": {"api_key": "sk-test"},
    })


def test_binding_returns_port_and_no_pool_model_on_coding(monkeypatch):
    monkeypatch.setattr(main, "config", _pooled_cfg())
    provider, sampling, port, selected = main._binding_for_request(_fake_request(8000))
    assert port == 8000
    assert provider.name == "deepseek"
    assert selected is None  # 无 pool


def test_binding_selects_pool_model_on_writing(monkeypatch):
    monkeypatch.setattr(main, "config", _pooled_cfg())
    provider, sampling, port, selected = main._binding_for_request(_fake_request(8001))
    assert port == 8001
    assert selected in (
        "deepseek-v4-flash", "deepseek-v4-pro", "mimo-v2.5", "mimo-v2.5-pro",
    )
    # 选中的 provider 必须与选中的 model 一致
    assert selected in (provider.flash_model, provider.pro_model)


def test_binding_pool_provider_varies_with_pick(monkeypatch):
    """多次调用应能同时出现 deepseek 与 mimo（逐请求重掷，跨家族）。"""
    monkeypatch.setattr(main, "config", _pooled_cfg())
    names = {main._binding_for_request(_fake_request(8001))[0].name for _ in range(200)}
    assert names == {"deepseek", "mimo"}
