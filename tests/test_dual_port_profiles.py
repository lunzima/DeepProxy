"""测试双端口绑定 + 强制覆盖采样参数。

- coding_port → precise_sampling profile
- writing_port → creative_sampling profile
- 客户端在请求体里给的 4 个采样参数（temperature/top_p/penalties）被覆盖
"""
from __future__ import annotations

import pytest

from deep_proxy.config import (
    CreativeSamplingConfig,
    DeepSeekConfig,
    PreciseSamplingConfig,
    ProxyConfig,
)
from deep_proxy.router import DeepProxyRouter


@pytest.fixture
def router():
    cfg = ProxyConfig(deepseek=DeepSeekConfig(api_key="sk"))
    return DeepProxyRouter(cfg)


class TestProxyConfigPorts:
    def test_default_dual_ports(self):
        cfg = ProxyConfig()
        assert cfg.coding_port == 8000
        assert cfg.writing_port == 8001

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("PROXY_CODING_PORT", "9000")
        monkeypatch.setenv("PROXY_WRITING_PORT", "9001")
        cfg = ProxyConfig.from_env()
        assert cfg.coding_port == 9000
        assert cfg.writing_port == 9001


class TestForcedOverride:
    """sampling_profile 提供时，4 个采样参数强制覆盖客户端值。"""

    async def test_precise_profile_overrides_client_temperature(self, router):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
            # 客户端给了离谱值
            "temperature": 1.99,
            "top_p": 0.5,
            "presence_penalty": -1.5,
            "frequency_penalty": 1.5,
        }
        p = await router.prepare_request(body, sampling_profile=PreciseSamplingConfig())
        # 客户端值被覆盖：精确 profile temperature 在 [0.25, 0.45]
        assert 0.25 <= p["temperature"] <= 0.45
        # top_p 固定 0.95
        assert p["top_p"] == 0.95
        # penalties 固定 0
        assert p["presence_penalty"] == 0.0
        assert p["frequency_penalty"] == 0.0

    async def test_creative_profile_overrides_client_temperature(self, router):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.0,
            "top_p": 0.1,
        }
        p = await router.prepare_request(body, sampling_profile=CreativeSamplingConfig())
        # 客户端 0.0 被覆盖到 [0.90, 1.20] 区间
        rp = CreativeSamplingConfig()
        assert rp.temperature_min <= p["temperature"] <= rp.temperature_max
        assert rp.top_p_min <= p["top_p"] <= rp.top_p_max

    async def test_no_profile_falls_back_to_setdefault(self, router):
        """sampling_profile=None 时退回 legacy default 行为（setdefault）。"""
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.42,  # 显式给值
        }
        p = await router.prepare_request(body)
        # 客户端值保留（setdefault 语义，未被覆盖）
        assert p["temperature"] == 0.42

    async def test_precise_profile_forced_even_without_client_values(self, router):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body, sampling_profile=PreciseSamplingConfig())
        assert 0.25 <= p["temperature"] <= 0.45
        assert p["top_p"] == 0.95


class TestPortToProfileMapping:
    """main.py 端口检测助手 _profile_for_request 的逻辑（不起 server，只测分派）。"""

    def test_profile_mapping_is_correct(self):
        from deep_proxy import main as m

        cfg = ProxyConfig(
            deepseek=DeepSeekConfig(api_key="sk"),
            coding_port=8000,
            writing_port=8001,
        )

        # mock 一个最小 Request
        class _Req:
            def __init__(self, port):
                self.scope = {"server": ("127.0.0.1", port)}

        # 将全局 config 临时替换
        old = m.config
        try:
            m.config = cfg
            assert m._profile_for_request(_Req(8000)) is cfg.precise_sampling
            assert m._profile_for_request(_Req(8001)) is cfg.creative_sampling
            # 未配置端口 → None
            assert m._profile_for_request(_Req(9999)) is None
            # 无 scope.server → None
            class _BadReq:
                scope = {}
            assert m._profile_for_request(_BadReq()) is None
        finally:
            m.config = old
