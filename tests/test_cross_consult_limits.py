"""Cross-Consult 限额测试：quota（consult 不再设武断的输入字符上限）。"""
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


def _text_chunks(text: str):
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"},
                      "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": text},
                      "finish_reason": "stop"}]},
    ]


async def test_quota_exhausted_returns_error_tool_result(cfg_with_low_quota):
    """max_calls_per_request=1，agent 第二次调 cross_consult 时应收到 quota 错误。

    重发走流式聚合：第 1 次重发吐 tc2，第 2 次重发吐 "final"。
    """
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_with_low_quota)
    provider = cfg_with_low_quota.providers["deepseek"]

    # 主响应初始（非流式）：返回 tc1
    initial = _tc_response("tc1", {"question": "q1"})

    # 重发流式：第 1 次吐 tc2（带 cross_consult tool_call）、第 2 次吐 "final"
    def resend_factory(config, body, *, provider=None):
        # 检测重发轮次：tool 消息的数量
        tool_count = sum(1 for m in body.get("messages", []) if m.get("role") == "tool")
        if tool_count == 1:
            # 第一次重发：吐 tc2 stream（带 cross_consult tool_call）
            async def gen():
                yield {"choices": [{"index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None}]}
                yield {"choices": [{"index": 0,
                                    "delta": {"tool_calls": [{
                                        "index": 0, "id": "tc2", "type": "function",
                                        "function": {"name": "cross_consult",
                                                     "arguments": json.dumps({"question": "q2"})},
                                    }]},
                                    "finish_reason": None}]}
                yield {"choices": [{"index": 0, "delta": {},
                                    "finish_reason": "tool_calls"}]}
            return gen()

        async def gen():
            for c in _text_chunks("final"):
                yield c
        return gen()

    # executor 应仅被调一次（第二次被 quota 拦截）
    executor_mock = AsyncMock(return_value="external 1")

    # 初始调用经 aggregate_stream_to_response（cc 活跃时 router.chat_completions
    # 走流式聚合以获取 chunk 级超时保护）；重发仍走 iter_litellm_chunks 流式路径。
    with patch("deep_proxy.router.aggregate_stream_to_response",
               new=AsyncMock(return_value=initial)), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult",
               new=executor_mock), \
         patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=resend_factory):
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
