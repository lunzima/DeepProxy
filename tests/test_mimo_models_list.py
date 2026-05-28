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


def test_mimo_models_constants():
    from deep_proxy.mimo_models import MIMO_FLASH, MIMO_PRO, MIMO_MODELS
    assert MIMO_FLASH == "mimo-v2.5"
    assert MIMO_PRO == "mimo-v2.5-pro"
    assert MIMO_FLASH in MIMO_MODELS
    assert MIMO_PRO in MIMO_MODELS


def test_mimo_models_have_metadata():
    from deep_proxy.mimo_models import MIMO_MODELS
    flash = MIMO_MODELS["mimo-v2.5"]
    assert flash["id"] == "mimo-v2.5"
    assert flash["owned_by"] == "xiaomi"


def test_build_models_list_for_mimo_provider(provider_mimo):
    from deep_proxy.models_list import build_models_list
    models = build_models_list(raw=[], provider=provider_mimo)
    ids = {m["id"] for m in models}
    assert "mimo-v2.5" in ids
    assert "mimo-v2.5-pro" in ids
    # MiMo 模型条目不应含 deepseek owned_by
    for m in models:
        if m["id"].startswith("mimo-"):
            assert m["owned_by"] == "xiaomi"


def test_build_models_list_for_mimo_has_correct_context_window(provider_mimo):
    from deep_proxy.models_list import build_models_list
    models = build_models_list(raw=[], provider=provider_mimo)
    flash = next(m for m in models if m["id"] == "mimo-v2.5")
    assert flash["context_length"] == 1_000_000
    assert flash["max_completion_tokens"] == 128_000  # MiMo 上限，不是 DeepSeek 的 384K


def test_build_models_list_for_deepseek_unchanged(provider_deepseek):
    """传 provider_deepseek 时行为应与不传 provider 一致。"""
    from deep_proxy.models_list import build_models_list
    old = build_models_list(raw=[])
    new = build_models_list(raw=[], provider=provider_deepseek)
    assert {m["id"] for m in old} == {m["id"] for m in new}
