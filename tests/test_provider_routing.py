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


def test_proxyconfig_provider_for_port():
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "coding_port": 8000,
        "writing_port": 8001,
        "deepseek": {"api_key": "sk"},
    }))
    p_coding = cfg.provider_for_port(8000)
    assert p_coding is not None
    assert p_coding.name == "deepseek"
    p_writing = cfg.provider_for_port(8001)
    assert p_writing.name == "deepseek"
    assert cfg.provider_for_port(9999) is None


def test_proxyconfig_sampling_profile_for_port():
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "coding_port": 8000,
        "writing_port": 8001,
        "deepseek": {"api_key": "sk"},
    }))
    sp = cfg.sampling_profile_for_port(8000)
    assert sp is cfg.precise_sampling
    sw = cfg.sampling_profile_for_port(8001)
    assert sw is cfg.creative_sampling
    assert cfg.sampling_profile_for_port(9999) is None


def test_assemble_litellm_body_uses_provider_when_given():
    from deep_proxy.config import ProxyConfig, normalize_legacy_config
    from deep_proxy.providers import Provider
    from deep_proxy.litellm_client import _assemble_litellm_body

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk-default", "api_base": "https://api.deepseek.com"},
    }))
    mimo = Provider(
        name="mimo",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        api_key="tp-mimo",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
    )
    body = {"model": "mimo-v2.5", "messages": [{"role": "user", "content": "hi"}]}
    call_body = _assemble_litellm_body(body, cfg, provider=mimo)
    assert call_body["api_key"] == "tp-mimo"
    assert call_body["api_base"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert call_body["model"] == "openai/mimo-v2.5"


def test_assemble_litellm_body_falls_back_to_deepseek_when_no_provider():
    from deep_proxy.config import ProxyConfig, normalize_legacy_config
    from deep_proxy.litellm_client import _assemble_litellm_body

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk-default", "api_base": "https://api.deepseek.com"},
    }))
    body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
    call_body = _assemble_litellm_body(body, cfg, provider=None)
    assert call_body["api_key"] == "sk-default"
    assert call_body["model"] == "deepseek/deepseek-v4-flash"


def test_provider_for_port_dual_config(cfg_dual):
    assert cfg_dual.provider_for_port(8000).name == "deepseek"
    assert cfg_dual.provider_for_port(8001).name == "mimo"
    assert cfg_dual.provider_for_port(9999) is None


def test_sampling_profile_for_port_dual(cfg_dual):
    sp = cfg_dual.sampling_profile_for_port(8000)
    assert sp is cfg_dual.precise_sampling
    sw = cfg_dual.sampling_profile_for_port(8001)
    assert sw is cfg_dual.creative_sampling


def test_assemble_litellm_body_routes_mimo_extras_to_extra_body():
    """MiMo 的 reasoning_effort / thinking 应放进 extra_body 透传，不留在顶层。"""
    from deep_proxy.config import ProxyConfig, normalize_legacy_config
    from deep_proxy.providers import Provider
    from deep_proxy.litellm_client import _assemble_litellm_body

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    mimo = Provider(
        name="mimo",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        api_key="tp-test",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
        allowed_extra_params=["reasoning_effort", "thinking"],
    )
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
    }
    call_body = _assemble_litellm_body(body, cfg, provider=mimo)
    # 顶层不再出现这些字段
    assert "reasoning_effort" not in call_body
    assert "thinking" not in call_body
    # 它们落在 extra_body 里
    assert call_body["extra_body"] == {
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
    }
    # allowed_openai_params 不再被注入
    assert "allowed_openai_params" not in call_body


def test_assemble_litellm_body_no_extra_body_for_deepseek():
    """DeepSeek provider 的 allowed_extra_params 默认空，不应注入 extra_body。"""
    from deep_proxy.config import ProxyConfig, normalize_legacy_config
    from deep_proxy.providers import Provider
    from deep_proxy.litellm_client import _assemble_litellm_body

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    ds = Provider(
        name="deepseek",
        api_base="https://api.deepseek.com",
        api_key="sk-x",
        litellm_prefix="deepseek/",
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
    )
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "reasoning_effort": "max"},
    }
    call_body = _assemble_litellm_body(body, cfg, provider=ds)
    # DeepSeek path: thinking 保留在顶层（deepseek provider 原生接受）
    assert call_body["thinking"] == {"type": "enabled", "reasoning_effort": "max"}
    assert "extra_body" not in call_body


async def test_prepare_request_force_mimo_model_when_client_sends_alien_name(
    router_dual, provider_mimo,
):
    """spec §7: 写作 port 上无论客户端传什么 model 名，都应映射到 provider.flash_model。"""
    body = {
        "model": "deepseek-chat",  # alien for MiMo provider
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    # MiMo provider 必须看到自家模型名，不能是 deepseek-v4-flash
    assert out["model"] in (provider_mimo.flash_model, provider_mimo.pro_model)


async def test_prepare_request_force_mimo_model_when_client_sends_claude_name(
    router_dual, provider_mimo,
):
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    assert out["model"] in (provider_mimo.flash_model, provider_mimo.pro_model)


async def test_prepare_request_force_mimo_model_when_client_sends_gpt_name(
    router_dual, provider_mimo,
):
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    assert out["model"] in (provider_mimo.flash_model, provider_mimo.pro_model)


async def test_prepare_request_preserves_mimo_own_model_name(
    router_dual, provider_mimo,
):
    """已经是 provider 自家模型名时保持不变。"""
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    assert out["model"] == "mimo-v2.5"


async def test_prepare_request_preserves_mimo_pro_name(
    router_dual, provider_mimo,
):
    body = {
        "model": "mimo-v2.5-pro",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    assert out["model"] == "mimo-v2.5-pro"


async def test_prepare_request_deepseek_provider_still_normalizes_legacy_alias(
    router_dual, provider_deepseek,
):
    """DeepSeek 路径行为不变：legacy alias 仍走 normalize_model_name → v4-flash。"""
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.precise_sampling, provider=provider_deepseek,
    )
    # DeepSeek 路径 deepseek-chat 应被规范化为 v4-flash
    assert out["model"] == "deepseek-v4-flash"
