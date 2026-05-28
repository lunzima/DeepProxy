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
