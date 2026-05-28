"""Cross-Consult 限额测试：quota、max_input_chars。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def cfg_with_low_quota():
    from deep_proxy.config import ProxyConfig
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "deepseek/", "flash_model": "deepseek-v4-flash",
                         "pro_model": "deepseek-v4-pro"},
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
                     "pro_model": "mimo-v2.5-pro"},
        },
        "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
        "deepseek": {"api_key": "y"},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
            "max_calls_per_request": 1,
            "max_input_chars": 100,
        },
    })


def _tc_response(tcid: str, args: dict):
    return {
        "choices": [{
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": tcid, "type": "function",
                    "function": {"name": "cross_consult", "arguments": json.dumps(args)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def _text_response(t: str):
    return {"choices": [{"message": {"role": "assistant", "content": t},
                          "finish_reason": "stop"}]}


async def test_quota_exhausted_returns_error_tool_result(cfg_with_low_quota):
    """max_calls_per_request=1，agent 第二次调 cross_consult 时应收到 quota 错误。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_with_low_quota)
    provider = cfg_with_low_quota.providers["deepseek"]

    # 主响应链：tc → consult → tc again → quota error → final
    main_responses = [
        _tc_response("tc1", {"question": "q1"}),
        _tc_response("tc2", {"question": "q2"}),
        _text_response("final"),
    ]

    async def fake_main(config, body, *, provider=None):
        return main_responses.pop(0)

    # executor 应仅被调一次（第二次被 quota 拦截）
    executor_mock = AsyncMock(return_value="external 1")

    with patch("deep_proxy.router.call_litellm", new=AsyncMock(side_effect=fake_main)), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=executor_mock):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "go"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_with_low_quota.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "final"
    assert executor_mock.call_count == 1  # 只调了一次

    # tool_result 链中应能找到 quota error 字符串
    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert any("quota" in m["content"].lower() for m in tool_msgs)


async def test_input_too_long_returns_error_without_calling_executor(cfg_with_low_quota):
    """question + context > max_input_chars 时不调 executor，返回 error tool_result。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_with_low_quota)
    provider = cfg_with_low_quota.providers["deepseek"]

    long_q = "X" * 200  # exceeds max_input_chars=100

    main_responses = [
        _tc_response("tc1", {"question": long_q}),
        _text_response("got error"),
    ]

    async def fake_main(config, body, *, provider=None):
        return main_responses.pop(0)

    executor_mock = AsyncMock(return_value="should not be called")

    with patch("deep_proxy.router.call_litellm", new=AsyncMock(side_effect=fake_main)), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=executor_mock):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "go"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_with_low_quota.precise_sampling, provider=provider,
        )
        result = await router.chat_completions(body, provider=provider)

    assert result["choices"][0]["message"]["content"] == "got error"
    assert executor_mock.call_count == 0  # 从未调用 executor

    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert any("too long" in m["content"].lower() for m in tool_msgs)
