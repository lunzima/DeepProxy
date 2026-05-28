"""Provider / Port 配置模型测试。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_proxy.providers import Provider, PortBinding


def test_provider_minimal_fields():
    p = Provider(
        name="deepseek",
        api_base="https://api.deepseek.com",
        api_key="sk-test",
        litellm_prefix="deepseek/",
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
    )
    assert p.name == "deepseek"
    assert p.has_reasoning_content is True
    assert p.has_thinking_param is True
    assert p.reasoning_effort_field == "thinking.reasoning_effort"
    assert p.reasoning_effort_value == "max"
    assert p.max_output_tokens == 384000
    assert p.context_window == 1000000


def test_provider_mimo_overrides():
    p = Provider(
        name="mimo",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        api_key="tp-test",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
        reasoning_effort_field="reasoning_effort",
        reasoning_effort_value="high",
        max_output_tokens=128000,
    )
    assert p.reasoning_effort_field == "reasoning_effort"
    assert p.reasoning_effort_value == "high"
    assert p.max_output_tokens == 128000


def test_port_binding_minimal():
    b = PortBinding(port=8000, provider="deepseek", sampling="precise")
    assert b.port == 8000
    assert b.sampling == "precise"


def test_port_binding_invalid_sampling():
    with pytest.raises(ValidationError):
        PortBinding(port=8000, provider="deepseek", sampling="random")


from deep_proxy.config import ProxyConfig, normalize_legacy_config


def test_proxyconfig_with_providers_and_ports():
    cfg = ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek",
                "api_base": "https://api.deepseek.com",
                "api_key": "sk-test",
                "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash",
                "pro_model": "deepseek-v4-pro",
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
            },
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "sk-legacy-compat"},
    })
    assert "deepseek" in cfg.providers
    assert "mimo" in cfg.providers
    assert len(cfg.ports) == 2
    assert cfg.ports[0].provider == "deepseek"


def test_normalize_legacy_config_no_providers_field():
    raw = {
        "coding_port": 8000,
        "writing_port": 8001,
        "deepseek": {
            "api_key": "sk-legacy",
            "api_base": "https://api.deepseek.com",
        },
    }
    normalized = normalize_legacy_config(raw)
    assert "providers" in normalized
    assert "deepseek" in normalized["providers"]
    assert normalized["providers"]["deepseek"]["api_key"] == "sk-legacy"
    assert "ports" in normalized
    assert len(normalized["ports"]) == 2
    assert normalized["ports"][0]["port"] == 8000
    assert normalized["ports"][0]["provider"] == "deepseek"
    assert normalized["ports"][0]["sampling"] == "precise"
    assert normalized["ports"][1]["port"] == 8001
    assert normalized["ports"][1]["sampling"] == "creative"


def test_normalize_legacy_config_passthrough_when_new_format():
    raw = {
        "providers": {"deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                                    "litellm_prefix": "deepseek/", "flash_model": "a", "pro_model": "b"}},
        "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
    }
    normalized = normalize_legacy_config(raw)
    assert normalized is raw  # 已是新格式，原样返回


def test_normalize_legacy_config_rejects_providers_without_ports():
    raw = {"providers": {"x": {"name": "x", "api_base": "y", "api_key": "z",
                               "litellm_prefix": "deepseek/", "flash_model": "a", "pro_model": "b"}}}
    with pytest.raises(ValueError, match="缺少 ports"):
        normalize_legacy_config(raw)


def test_normalize_legacy_config_rejects_ports_without_providers():
    raw = {"ports": [{"port": 8000, "provider": "x", "sampling": "precise"}]}
    with pytest.raises(ValueError, match="缺少 providers"):
        normalize_legacy_config(raw)
