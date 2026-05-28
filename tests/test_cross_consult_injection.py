"""Cross-Consult 请求路径注入测试。"""
from __future__ import annotations

import pytest


def test_tool_schema_basic_shape():
    from deep_proxy.cross_consult.schema import build_tool_schema
    s = build_tool_schema(tool_name="cross_consult")
    assert s["type"] == "function"
    assert s["function"]["name"] == "cross_consult"
    assert "description" in s["function"]
    params = s["function"]["parameters"]
    assert params["type"] == "object"
    assert "question" in params["properties"]
    assert "context" in params["properties"]
    assert "purpose" in params["properties"]
    assert params["required"] == ["question"]


def test_tool_schema_purpose_enum():
    from deep_proxy.cross_consult.schema import build_tool_schema
    s = build_tool_schema(tool_name="cross_consult")
    purpose = s["function"]["parameters"]["properties"]["purpose"]
    assert "enum" in purpose
    assert set(purpose["enum"]) == {
        "second_opinion", "cross_domain_help", "style_check", "logic_check", "other",
    }


def test_system_prompt_addendum_mentions_quota():
    from deep_proxy.cross_consult.schema import build_system_prompt_addendum
    text = build_system_prompt_addendum(tool_name="cross_consult", max_calls=3)
    assert "cross_consult" in text
    assert "3" in text
    assert "self-contained" in text


def test_system_prompt_addendum_uses_custom_tool_name():
    from deep_proxy.cross_consult.schema import build_system_prompt_addendum
    text = build_system_prompt_addendum(tool_name="ask_alt", max_calls=5)
    assert "ask_alt" in text
    assert "5" in text


async def test_prepare_request_injects_cross_consult_tool_when_enabled():
    """启用 + pair 存在时，tools 数组应包含 cross_consult。"""
    from deep_proxy.config import ProxyConfig
    from deep_proxy.router import DeepProxyRouter

    cfg = ProxyConfig.model_validate({
        "providers": {
            "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "deepseek/", "flash_model": "deepseek-v4-flash",
                         "pro_model": "deepseek-v4-pro"},
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
                     "pro_model": "mimo-v2.5-pro"},
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "y"},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
        },
    })
    router = DeepProxyRouter(cfg)
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling,
        provider=cfg.providers["deepseek"],
    )
    tools = out.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    assert "cross_consult" in names

    # system prompt 应被追加增量
    system_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert any("cross_consult" in m.get("content", "") for m in system_msgs)


async def test_prepare_request_skips_injection_when_disabled():
    from deep_proxy.config import ProxyConfig, normalize_legacy_config
    from deep_proxy.router import DeepProxyRouter

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    # cross_consult disabled by default
    router = DeepProxyRouter(cfg)
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling,
        provider=cfg.providers["deepseek"],
    )
    tools = out.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    assert "cross_consult" not in names


async def test_prepare_request_skips_injection_when_pair_missing():
    """enabled=True 但当前 provider 在 pairs 中无对偶 → 不注入。"""
    from deep_proxy.config import ProxyConfig
    from deep_proxy.router import DeepProxyRouter

    cfg = ProxyConfig.model_validate({
        "providers": {
            "lonely": {"name": "lonely", "api_base": "x", "api_key": "y",
                       "litellm_prefix": "openai/", "flash_model": "a", "pro_model": "b"},
        },
        "ports": [{"port": 9000, "provider": "lonely", "sampling": "precise"}],
        "deepseek": {"api_key": "y"},
        "cross_consult": {"enabled": True, "pairs": {"deepseek": "mimo"}},  # no entry for 'lonely'
    })
    router = DeepProxyRouter(cfg)
    body = {"model": "a", "messages": [{"role": "user", "content": "hi"}]}
    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling,
        provider=cfg.providers["lonely"],
    )
    tools = out.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    assert "cross_consult" not in names


async def test_prepare_request_skips_injection_for_recursion_sentinel():
    """body 带 _deepproxy_cross_consult_internal=True 时，必须跳过注入。"""
    from deep_proxy.config import ProxyConfig
    from deep_proxy.router import DeepProxyRouter

    cfg = ProxyConfig.model_validate({
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
        "cross_consult": {"enabled": True, "pairs": {"deepseek": "mimo", "mimo": "deepseek"}},
    })
    router = DeepProxyRouter(cfg)
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "_deepproxy_cross_consult_internal": True,
    }
    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling,
        provider=cfg.providers["deepseek"],
    )
    tools = out.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    assert "cross_consult" not in names


async def test_prepare_request_appends_to_existing_tools_array():
    """客户端已传 tools 数组时，cross_consult 应 append，不替换。"""
    from deep_proxy.config import ProxyConfig
    from deep_proxy.router import DeepProxyRouter

    cfg = ProxyConfig.model_validate({
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
        "cross_consult": {"enabled": True, "pairs": {"deepseek": "mimo", "mimo": "deepseek"}},
    })
    router = DeepProxyRouter(cfg)
    user_tool = {
        "type": "function",
        "function": {"name": "user_thing", "parameters": {"type": "object", "properties": {}}},
    }
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [user_tool],
    }
    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling,
        provider=cfg.providers["deepseek"],
    )
    names = [t["function"]["name"] for t in out["tools"]]
    assert "user_thing" in names
    assert "cross_consult" in names
