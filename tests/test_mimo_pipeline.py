"""MiMo 路径下的协议差异适配测试。"""
from __future__ import annotations

from deep_proxy.compatibility.mimo_fixes import (
    inject_top_level_reasoning_effort,
)


def test_inject_top_level_reasoning_effort_no_thinking_no_existing():
    body = {"model": "mimo-v2.5", "messages": []}
    inject_top_level_reasoning_effort(body, value="high")
    assert body["reasoning_effort"] == "high"
    assert "thinking" not in body


def test_inject_skips_when_thinking_disabled():
    body = {"model": "mimo-v2.5", "messages": [], "thinking": {"type": "disabled"}}
    inject_top_level_reasoning_effort(body, value="high")
    # 显式 disabled 时不注入
    assert "reasoning_effort" not in body


def test_inject_respects_existing_reasoning_effort():
    body = {"model": "mimo-v2.5", "messages": [], "reasoning_effort": "low"}
    inject_top_level_reasoning_effort(body, value="high")
    # 客户端已显式设置时不覆盖
    assert body["reasoning_effort"] == "low"


import pytest


async def test_prepare_request_mimo_injects_top_level_reasoning_effort(
    router_dual, provider_mimo,
):
    """MiMo 路径：reasoning_effort 注入到顶层（不是 thinking 子字段）。"""
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "hello"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    assert out.get("reasoning_effort") == "high"
    # MiMo 路径不应注入 thinking.reasoning_effort
    thinking = out.get("thinking") or {}
    assert "reasoning_effort" not in thinking


async def test_prepare_request_deepseek_keeps_thinking_reasoning_effort(
    router_dual, provider_deepseek,
):
    """DeepSeek 路径行为不变：reasoning_effort 在 thinking 子字段。"""
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.precise_sampling, provider=provider_deepseek,
    )
    thinking = out.get("thinking", {})
    assert thinking.get("reasoning_effort") == "max"
    # 顶层 reasoning_effort 不应出现
    assert "reasoning_effort" not in out


async def test_prepare_request_mimo_disable_thinking_skips_reasoning_effort(
    router_dual, provider_mimo,
):
    """MiMo + thinking.type=disabled：不注入 reasoning_effort。"""
    body = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "disabled"},
    }
    out = await router_dual.prepare_request(
        body, sampling_profile=router_dual.config.creative_sampling, provider=provider_mimo,
    )
    assert "reasoning_effort" not in out
