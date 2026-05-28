"""Cross-Consult executor 单测（mock litellm 调用）。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.cross_consult.executor import execute_consult
from deep_proxy.providers import Provider


@pytest.fixture
def cfg_for_executor():
    return ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))


@pytest.fixture
def target_provider():
    return Provider(
        name="mimo",
        api_base="https://x",
        api_key="tp",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
        allowed_extra_params=["reasoning_effort", "thinking"],
    )


async def test_execute_consult_calls_target_pro_model(cfg_for_executor, target_provider):
    """consult 调用必须指向 target_provider.pro_model，不是 flash_model。"""
    fake_response = {"choices": [{"message": {"content": "external answer"}}]}
    with patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(return_value=fake_response)) as mocked:
        out = await execute_consult(
            question="What is 2+2?",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
        assert out == "external answer"
        call_args = mocked.call_args
        body = call_args.args[1]
        assert body["model"] == "mimo-v2.5-pro"


async def test_execute_consult_sets_recursion_sentinel(cfg_for_executor, target_provider):
    """consult 调用 body 必须带 _deepproxy_cross_consult_internal=True（防递归注入）。"""
    fake_response = {"choices": [{"message": {"content": "x"}}]}
    with patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(return_value=fake_response)) as mocked:
        await execute_consult(
            question="hi",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
        body = mocked.call_args.args[1]
        assert body.get("_deepproxy_cross_consult_internal") is True


async def test_execute_consult_no_tools_in_body(cfg_for_executor, target_provider):
    """consult 调用绝不携带 tools 数组（防递归）。"""
    fake_response = {"choices": [{"message": {"content": "x"}}]}
    with patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(return_value=fake_response)) as mocked:
        await execute_consult(
            question="hi",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
        body = mocked.call_args.args[1]
        assert "tools" not in body
        assert "tool_choice" not in body


async def test_execute_consult_includes_context_when_given(cfg_for_executor, target_provider):
    """context 不为空时应附加到 user message。"""
    captured_body = None

    async def fake_call(*args, **kwargs):
        nonlocal captured_body
        captured_body = args[1]
        return {"choices": [{"message": {"content": "ok"}}]}

    with patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(side_effect=fake_call)):
        await execute_consult(
            question="why?",
            context="some background",
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    user_msg = captured_body["messages"][-1]["content"]
    assert "why?" in user_msg
    assert "some background" in user_msg


async def test_execute_consult_returns_error_string_on_timeout(cfg_for_executor, target_provider):
    """超时返回带 [DeepProxy cross_consult error] 前缀的错误字符串，不抛异常。"""
    import asyncio

    async def slow_call(*args, **kwargs):
        await asyncio.sleep(2.0)
        return {"choices": [{"message": {"content": "should not see"}}]}

    with patch("deep_proxy.cross_consult.executor.call_litellm",
               new=AsyncMock(side_effect=slow_call)):
        out = await execute_consult(
            question="hi",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(
                enabled=True, call_timeout_seconds=1,
            ),
        )
    assert out.startswith("[DeepProxy cross_consult error]")
    assert "timeout" in out.lower() or "超时" in out
