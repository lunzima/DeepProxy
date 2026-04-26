"""测试 list_models 配置开关。"""
from __future__ import annotations

import pytest

from deep_proxy.config import DeepSeekConfig, ProxyConfig
from deep_proxy.router import DeepProxyRouter


async def test_default_only_v4(router: DeepProxyRouter):
    res = await router.list_models()
    ids = {m["id"] for m in res["data"]}
    assert "deepseek-v4-flash" in ids
    assert "deepseek-v4-pro" in ids
    assert "deepseek-chat" not in ids
    assert "deepseek-reasoner" not in ids


async def test_expose_legacy_includes_old_names():
    cfg = ProxyConfig(
        deepseek=DeepSeekConfig(api_key="sk", expose_legacy_models=True)
    )
    r = DeepProxyRouter(cfg)
    res = await r.list_models()
    ids = {m["id"] for m in res["data"]}
    assert "deepseek-v4-flash" in ids
    assert "deepseek-chat" in ids
    assert "deepseek-reasoner" in ids


async def test_custom_route_appears_once():
    from deep_proxy.config import ModelRoute
    cfg = ProxyConfig(
        deepseek=DeepSeekConfig(api_key="sk"),
        model_routes=[ModelRoute(model_name="my-alias", provider_model="deepseek-v4-flash")],
    )
    r = DeepProxyRouter(cfg)
    res = await r.list_models()
    ids = [m["id"] for m in res["data"]]
    assert ids.count("my-alias") == 1
