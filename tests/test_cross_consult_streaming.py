"""aggregate_stream_to_response 单测：首 chunk（TTFT/prefill）预算与 inter-chunk
idle 预算分离的回归覆盖。

背景：cross_consult 重发携带最大上下文（原对话 + 注入的 tool_result），其 prefill +
推理预热的 time-to-first-chunk 可能远超 inter-chunk idle 预算。历史 bug：单一 30s
预算同时守护首 chunk 与 chunk 间隙，导致健康上游在 prefill 阶段被误杀。
"""
from __future__ import annotations

import asyncio

import pytest

from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.cross_consult.streaming import aggregate_stream_to_response


@pytest.fixture
def cfg():
    return ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))


async def test_first_chunk_slower_than_idle_budget_is_allowed(cfg):
    """首 chunk 的 prefill 慢于 inter-chunk idle 预算，但在 first_chunk_timeout 内
    到达——不应超时。"""
    async def slow_first(config, body, *, provider=None):
        await asyncio.sleep(0.25)  # > idle_timeout(0.1)，< first_chunk_timeout(2.0)
        yield {"choices": [{"index": 0,
                            "delta": {"content": "prefilled"},
                            "finish_reason": "stop"}]}

    result = await aggregate_stream_to_response(
        cfg, {}, provider=None,
        idle_timeout=0.1, first_chunk_timeout=2.0, iter_fn=slow_first,
    )
    assert "_dp_error" not in result
    assert result["choices"][0]["message"]["content"] == "prefilled"


async def test_mid_stream_gap_trips_idle_timeout(cfg):
    """首 chunk 立即到达后，chunk 间隙超过 idle_timeout——按 inter-chunk idle 超时
    （phase=mid_stream），不享受 first_chunk 宽限。"""
    async def stall_after_first(config, body, *, provider=None):
        yield {"choices": [{"index": 0,
                            "delta": {"content": "first"},
                            "finish_reason": None}]}
        await asyncio.sleep(0.5)  # > idle_timeout(0.1)
        yield {"choices": [{"index": 0,
                            "delta": {"content": "second"},
                            "finish_reason": "stop"}]}

    result = await aggregate_stream_to_response(
        cfg, {}, provider=None,
        idle_timeout=0.1, first_chunk_timeout=2.0, iter_fn=stall_after_first,
    )
    assert "_dp_error" in result
    assert "mid_stream" in result["_dp_error"]


async def test_first_chunk_timeout_trips_when_prefill_exceeds_budget(cfg):
    """首 chunk 连 first_chunk_timeout 都超了——按 phase=first_chunk 超时。"""
    async def never_first(config, body, *, provider=None):
        await asyncio.sleep(0.5)  # > first_chunk_timeout(0.1)
        yield {"choices": [{"index": 0,
                            "delta": {"content": "too late"},
                            "finish_reason": "stop"}]}

    result = await aggregate_stream_to_response(
        cfg, {}, provider=None,
        idle_timeout=2.0, first_chunk_timeout=0.1, iter_fn=never_first,
    )
    assert "_dp_error" in result
    assert "first_chunk" in result["_dp_error"]


async def test_first_chunk_timeout_defaults_to_idle_when_unset(cfg):
    """未传 first_chunk_timeout 时退回 idle_timeout 守护首 chunk（向后兼容）。"""
    async def slow_first(config, body, *, provider=None):
        await asyncio.sleep(0.3)  # > idle_timeout(0.1)
        yield {"choices": [{"index": 0,
                            "delta": {"content": "x"},
                            "finish_reason": "stop"}]}

    result = await aggregate_stream_to_response(
        cfg, {}, provider=None,
        idle_timeout=0.1, iter_fn=slow_first,
    )
    assert "_dp_error" in result
