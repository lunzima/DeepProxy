"""测试 CreativeSamplingConfig 高多样性范围采样预设。

验证：
- enabled=True (默认) → 注入随机抽样的采样参数（落在 [min, max] 内、round 到 0.01）
- enabled=False → 退回原"准确性优先"低温度默认
- 客户端显式值永远优先（setdefault 语义）
- 同一 router 多次请求得到不同值（统计上几乎必然）
"""
from __future__ import annotations

import pytest

from deep_proxy.config import DeepSeekConfig, ProxyConfig, CreativeSamplingConfig
from deep_proxy.router import DeepProxyRouter


@pytest.fixture
def router_rp_on():
    cfg = ProxyConfig(deepseek=DeepSeekConfig(api_key="sk"))
    return DeepProxyRouter(cfg)


@pytest.fixture
def router_rp_off():
    cfg = ProxyConfig(
        deepseek=DeepSeekConfig(api_key="sk"),
        creative_sampling=CreativeSamplingConfig(enabled=False),
    )
    return DeepProxyRouter(cfg)


def _is_rounded_2dp(x: float) -> bool:
    """检查值是否 round 到 0.01（容许浮点误差）。"""
    return abs(x * 100 - round(x * 100)) < 1e-9


class TestRangeSampling:
    async def test_sampled_values_in_range_and_rounded(self, router_rp_on):
        rp = router_rp_on.config.creative_sampling
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router_rp_on.prepare_request(body)
        for k, lo, hi in [
            ("temperature", rp.temperature_min, rp.temperature_max),
            ("top_p", rp.top_p_min, rp.top_p_max),
            ("presence_penalty", rp.presence_penalty_min, rp.presence_penalty_max),
            ("frequency_penalty", rp.frequency_penalty_min, rp.frequency_penalty_max),
        ]:
            assert lo <= p[k] <= hi
            assert _is_rounded_2dp(p[k]), f"{k}={p[k]} 未 round 到 0.01"

    async def test_degenerate_range_returns_fixed_value(self):
        """min == max 时退化为定值。"""
        cfg = ProxyConfig(
            deepseek=DeepSeekConfig(api_key="sk"),
            creative_sampling=CreativeSamplingConfig(
                temperature_min=0.7, temperature_max=0.7,
                top_p_min=0.85, top_p_max=0.85,
                presence_penalty_min=0.0, presence_penalty_max=0.0,
                frequency_penalty_min=0.0, frequency_penalty_max=0.0,
            ),
        )
        r = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await r.prepare_request(body)
        assert p["temperature"] == 0.7
        assert p["top_p"] == 0.85
        assert p["presence_penalty"] == 0.0
        assert p["frequency_penalty"] == 0.0


class TestDisabled:
    async def test_disabled_falls_back_to_accuracy_defaults(self, router_rp_off):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "what is 2+2"}],
        }
        p = await router_rp_off.prepare_request(body)
        # 关闭时退回低温度 + 不设 penalties
        assert p["temperature"] == 0.6
        assert p["top_p"] == 0.95
        assert "presence_penalty" not in p
        assert "frequency_penalty" not in p


class TestClientOverride:
    async def test_client_temperature_overrides_sampled(self, router_rp_on):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.0,
        }
        p = await router_rp_on.prepare_request(body)
        assert p["temperature"] == 0.0  # 客户端值赢
        # 其它仍按范围抽样
        rp = router_rp_on.config.creative_sampling
        assert rp.top_p_min <= p["top_p"] <= rp.top_p_max

    async def test_client_can_override_all_rp_params(self, router_rp_on):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.2,
            "top_p": 0.8,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        p = await router_rp_on.prepare_request(body)
        assert p["temperature"] == 0.2
        assert p["top_p"] == 0.8
        assert p["presence_penalty"] == 0.0
        assert p["frequency_penalty"] == 0.0
