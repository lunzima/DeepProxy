"""MiMo 模型列表与定价数据测试。"""
from __future__ import annotations


def test_mimo_pricing_self_contained():
    """mimo_pricing 模块不依赖 deepseek_pricing。"""
    import deep_proxy.mimo_pricing as m
    # 模块 import 列表里不应出现 deepseek_pricing
    import inspect
    src = inspect.getsource(m)
    assert "deepseek_pricing" not in src, "mimo_pricing 不允许 import deepseek_pricing（§3.1）"


def test_mimo_pricing_has_v25_entries():
    from deep_proxy.mimo_pricing import _MIMO_PRICING, _MIMO_CONTEXT_WINDOW, _MIMO_MAX_OUTPUT
    assert "mimo-v2.5" in _MIMO_PRICING
    assert "mimo-v2.5-pro" in _MIMO_PRICING
    flash = _MIMO_PRICING["mimo-v2.5"]
    assert flash["prompt"] > 0
    assert flash["completion"] > 0
    assert _MIMO_CONTEXT_WINDOW == 1_000_000
    assert _MIMO_MAX_OUTPUT == 128_000


def test_mimo_model_pricing_callable():
    from deep_proxy.mimo_pricing import model_pricing
    p = model_pricing("mimo-v2.5")
    assert "prompt" in p
    assert "completion" in p
    # 未知模型返回 {}（与 deepseek 同协议）
    assert model_pricing("unknown-model") == {}
