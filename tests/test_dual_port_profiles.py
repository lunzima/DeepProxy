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

    def test_bound_ports_follows_ports_declaration(self):
        """服务器绑定的端口 = ports[] 声明的端口，而非硬编码的 coding_port/writing_port。
        新格式把端口 remap 到 9000/9001 时，server 必须绑定 9000/9001（否则绑 8000/8001
        → provider_for_port 返回 None → 所有请求静默退回直通路径）。"""
        from deep_proxy.config import normalize_legacy_config

        base = {
            "providers": {"deepseek": {
                "name": "deepseek", "api_base": "x", "api_key": "y",
                "litellm_prefix": "deepseek/", "flash_model": "a", "pro_model": "b"}},
            "deepseek": {"api_key": "y"},
        }
        # ports remap 到 9000/9001，coding/writing_port 仍是默认 8000/8001
        cfg = ProxyConfig.model_validate(normalize_legacy_config({
            **base,
            "ports": [
                {"port": 9000, "provider": "deepseek", "sampling": "precise"},
                {"port": 9001, "provider": "deepseek", "sampling": "creative"},
            ],
        }))
        assert cfg.bound_ports() == [9000, 9001]

    def test_bound_ports_legacy_defaults(self):
        """老格式（无 providers/ports）经 normalize 后绑定 [8000, 8001]，与历史一致。"""
        from deep_proxy.config import normalize_legacy_config

        cfg = ProxyConfig.model_validate(normalize_legacy_config({"deepseek": {"api_key": "y"}}))
        assert cfg.bound_ports() == [8000, 8001]

    def test_bound_ports_single_port(self):
        """单端口部署：仅声明 8000 → server 只绑定 [8000]（不再凭空绑定幽灵 8001）。"""
        from deep_proxy.config import normalize_legacy_config

        cfg = ProxyConfig.model_validate(normalize_legacy_config({
            "providers": {"deepseek": {
                "name": "deepseek", "api_base": "x", "api_key": "y",
                "litellm_prefix": "deepseek/", "flash_model": "a", "pro_model": "b"}},
            "deepseek": {"api_key": "y"},
            "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
        }))
        assert cfg.bound_ports() == [8000]


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
    """main.py 端口检测助手 _binding_for_request 的逻辑（不起 server，只测分派）。"""

    def test_profile_mapping_is_correct(self):
        from deep_proxy import main as m
        from deep_proxy.config import normalize_legacy_config

        cfg = ProxyConfig.model_validate(normalize_legacy_config({
            "coding_port": 8000,
            "writing_port": 8001,
            "deepseek": {"api_key": "sk"},
        }))

        # mock 一个最小 Request
        class _Req:
            def __init__(self, port):
                self.scope = {"server": ("127.0.0.1", port)}

        # 将全局 config 临时替换
        old = m.config
        try:
            m.config = cfg
            _, sp_coding, port_coding, sel_coding = m._binding_for_request(_Req(8000))
            assert sp_coding is cfg.precise_sampling
            assert port_coding == 8000
            assert sel_coding is None  # 无 pool
            _, sp_writing, port_writing, sel_writing = m._binding_for_request(_Req(8001))
            assert sp_writing is cfg.creative_sampling
            assert port_writing == 8001
            assert sel_writing is None
            # 未配置端口 → (None, None, None, None)
            provider_none, sp_none, _, _ = m._binding_for_request(_Req(9999))
            assert provider_none is None
            assert sp_none is None
            # 无 scope.server → (None, None, None, None)
            class _BadReq:
                scope = {}
            provider_bad, sp_bad, port_bad, sel_bad = m._binding_for_request(_BadReq())
            assert provider_bad is None
            assert sp_bad is None
            assert port_bad is None
        finally:
            m.config = old
