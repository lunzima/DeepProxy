"""Writing-port 加权模型桶：config 校验 + 选择器。"""
from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from deep_proxy.config import ProxyConfig
from deep_proxy.pool import select_pool_target


def _base_providers() -> dict:
    return {
        "deepseek": {
            "name": "deepseek", "api_base": "https://api.deepseek.com",
            "api_key": "sk-test", "litellm_prefix": "deepseek/",
            "flash_model": "deepseek-v4-flash", "pro_model": "deepseek-v4-pro",
        },
        "mimo": {
            "name": "mimo", "api_base": "https://m/v1", "api_key": "tp-test",
            "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
            "pro_model": "mimo-v2.5-pro",
        },
    }


_FULL_POOL = [
    {"provider": "deepseek", "model": "deepseek-v4-flash", "weight": 1},
    {"provider": "mimo", "model": "mimo-v2.5", "weight": 1},
    {"provider": "deepseek", "model": "deepseek-v4-pro", "weight": 1},
    {"provider": "mimo", "model": "mimo-v2.5-pro", "weight": 1},
]


def _cfg_with_pool(pool: list[dict]) -> ProxyConfig:
    return ProxyConfig.model_validate({
        "providers": _base_providers(),
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative", "model_pool": pool},
        ],
        "deepseek": {"api_key": "sk-test"},
    })


def test_valid_pool_loads():
    cfg = _cfg_with_pool(_FULL_POOL)
    binding = next(b for b in cfg.ports if b.port == 8001)
    assert binding.model_pool is not None
    assert len(binding.model_pool) == 4


def test_pool_rejects_unknown_provider():
    bad = [{"provider": "anthropic", "model": "claude", "weight": 1}]
    with pytest.raises(ValidationError):
        _cfg_with_pool(bad)


def test_pool_rejects_model_not_flash_or_pro():
    bad = [{"provider": "deepseek", "model": "deepseek-v4-turbo", "weight": 1}]
    with pytest.raises(ValidationError):
        _cfg_with_pool(bad)


def test_pool_rejects_nonpositive_weight():
    bad = [{"provider": "deepseek", "model": "deepseek-v4-flash", "weight": 0}]
    with pytest.raises(ValidationError):
        _cfg_with_pool(bad)


def test_select_resolves_provider_and_model():
    cfg = _cfg_with_pool(_FULL_POOL)
    binding = next(b for b in cfg.ports if b.port == 8001)
    provider, model = select_pool_target(binding, cfg, rng=random.Random(0))
    assert provider.name in ("deepseek", "mimo")
    assert model in (
        "deepseek-v4-flash", "deepseek-v4-pro", "mimo-v2.5", "mimo-v2.5-pro",
    )
    # 选中的 model 必须属于选中的 provider
    assert model in (provider.flash_model, provider.pro_model)


def test_select_honors_weight_distribution():
    """权重 9:1 时高权重条目应占绝大多数。"""
    pool = [
        {"provider": "deepseek", "model": "deepseek-v4-flash", "weight": 9},
        {"provider": "mimo", "model": "mimo-v2.5", "weight": 1},
    ]
    cfg = _cfg_with_pool(pool)
    binding = next(b for b in cfg.ports if b.port == 8001)
    rng = random.Random(42)
    picks = [select_pool_target(binding, cfg, rng=rng)[1] for _ in range(1000)]
    flash_count = picks.count("deepseek-v4-flash")
    assert 850 < flash_count < 950  # ~90%


def test_select_equal_weights_roughly_uniform():
    cfg = _cfg_with_pool(_FULL_POOL)
    binding = next(b for b in cfg.ports if b.port == 8001)
    rng = random.Random(7)
    picks = [select_pool_target(binding, cfg, rng=rng)[1] for _ in range(2000)]
    for m in ("deepseek-v4-flash", "mimo-v2.5", "deepseek-v4-pro", "mimo-v2.5-pro"):
        assert 400 < picks.count(m) < 600  # ~25% each


# ---------------------------------------------------------------------------
# /v1/models 并集（pool 配置时列出池内 provider 家族并集）
# ---------------------------------------------------------------------------


async def test_list_models_union_across_pool_families():
    from deep_proxy.config import DeepSeekConfig
    from deep_proxy.providers import Provider
    from deep_proxy.router import DeepProxyRouter

    # deepseek api_key 置空 → fetch_upstream_models 立即返回 []，走本地兜底（无网络）
    cfg = ProxyConfig.model_validate({
        "providers": _base_providers(),
        "ports": [{"port": 8001, "provider": "mimo", "sampling": "creative",
                   "model_pool": _FULL_POOL}],
        "deepseek": {"api_key": ""},
    })
    router = DeepProxyRouter(cfg)
    home = cfg.providers["mimo"]
    deepseek = cfg.providers["deepseek"]
    res = await router.list_models(provider=home, pool_providers=[home, deepseek])
    ids = {m["id"] for m in res["data"]}
    assert "mimo-v2.5" in ids          # mimo 家族
    assert "deepseek-v4-flash" in ids  # deepseek 家族
    # 去重：每个 id 仅一次
    all_ids = [m["id"] for m in res["data"]]
    assert len(all_ids) == len(set(all_ids))
