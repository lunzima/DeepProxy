"""DeepProxyRouter 层的 per-port 动态阈值控制器接线。"""
from __future__ import annotations

from deep_proxy.config import ProxyConfig
from deep_proxy.optimization.dynamic_threshold import DynamicThresholdController
from deep_proxy.providers import Provider
from deep_proxy.router import DeepProxyRouter


def _cfg(dynamic_enabled: bool = True) -> ProxyConfig:
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek", "api_base": "https://api.deepseek.com",
                "api_key": "sk-test", "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash", "pro_model": "deepseek-v4-pro",
            },
            "mimo": {
                "name": "mimo", "api_base": "https://m/v1", "api_key": "tp-test",
                "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
                "pro_model": "mimo-v2.5-pro", "reasoning_effort_field": "reasoning_effort",
                "reasoning_effort_value": "high", "max_output_tokens": 128000,
                "allowed_extra_params": ["reasoning_effort", "thinking"],
            },
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "sk-test"},
        # 关闭优化以避免 prepare_request 走 compressor/skills 重路径
        "optimization": {"enabled": False},
        "flash_upgrade": {"enabled": True, "router_type": "rule",
                          "dynamic_threshold": {"enabled": dynamic_enabled}},
    })


def _mimo() -> Provider:
    return Provider(
        name="mimo", api_base="https://m/v1", api_key="tp-test",
        litellm_prefix="openai/", flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
        reasoning_effort_field="reasoning_effort", reasoning_effort_value="high",
        max_output_tokens=128000, allowed_extra_params=["reasoning_effort", "thinking"],
    )


def test_controller_created_per_port_when_enabled():
    router = DeepProxyRouter(_cfg(dynamic_enabled=True))
    c = router._controller_for_port(8001)
    assert isinstance(c, DynamicThresholdController)


def test_controller_none_when_disabled():
    router = DeepProxyRouter(_cfg(dynamic_enabled=False))
    assert router._controller_for_port(8001) is None


def test_controller_same_instance_per_port():
    router = DeepProxyRouter(_cfg(dynamic_enabled=True))
    assert router._controller_for_port(8001) is router._controller_for_port(8001)
    assert router._controller_for_port(8000) is not router._controller_for_port(8001)


async def test_prepare_request_records_threshold_decision():
    """flash 起始 + 传入 port → 该 port 控制器记录一次阈值决策。"""
    router = DeepProxyRouter(_cfg(dynamic_enabled=True))
    body = {"model": "mimo-v2.5", "messages": [{"role": "user", "content": "hi"}]}
    await router.prepare_request(
        body, sampling_profile=router.config.creative_sampling,
        provider=_mimo(), port=8001,
    )
    assert router._controller_for_port(8001).samples == 1


async def test_prepare_request_pro_pick_not_recorded():
    """pro 起始（pool 直接选中 pro）→ 不进控制器窗口。"""
    router = DeepProxyRouter(_cfg(dynamic_enabled=True))
    body = {"model": "mimo-v2.5-pro", "messages": [{"role": "user", "content": "hi"}]}
    await router.prepare_request(
        body, sampling_profile=router.config.creative_sampling,
        provider=_mimo(), port=8001,
    )
    assert router._controller_for_port(8001).samples == 0
    assert body["model"] == "mimo-v2.5-pro"  # pin 在 pro
