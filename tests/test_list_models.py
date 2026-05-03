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


# ---------------------------------------------------------------------------
# display_name / Anthropic 字段 / 字段优先级（外部 Agent 复审反馈覆盖）
# ---------------------------------------------------------------------------


def test_display_name_canonical_v4():
    from deep_proxy.models_list import _build_display_name
    assert _build_display_name("deepseek-v4-flash") == "DeepSeek V4 Flash"
    assert _build_display_name("deepseek-v4-pro") == "DeepSeek V4 Pro"


def test_display_name_preserves_bracket_suffix():
    """[1m] 后缀必须原样保留，不被 .capitalize() 破坏。"""
    from deep_proxy.models_list import _build_display_name
    assert _build_display_name("deepseek-v4-flash[1m]") == "DeepSeek V4 Flash [1m]"
    assert _build_display_name("deepseek-v4-pro[1m]") == "DeepSeek V4 Pro [1m]"


def test_display_name_vendor_overrides():
    """gpt-4o-mini → GPT 4o Mini（非 Gpt 4o Mini）；其它常见 vendor 段同样覆盖。"""
    from deep_proxy.models_list import _build_display_name
    assert _build_display_name("gpt-4o-mini") == "GPT 4o Mini"
    assert _build_display_name("claude-sonnet") == "Claude Sonnet"


def test_display_name_param_count_uppercase():
    """参数量段（72b / 405B / 1.5b）按业内惯例统一大写为 B 结尾。"""
    from deep_proxy.models_list import _build_display_name
    assert _build_display_name("qwen-72b") == "Qwen 72B"
    assert _build_display_name("llama-3.1-405b-instruct") == "Llama 3.1 405B Instruct"
    assert _build_display_name("mistral-1.5b") == "Mistral 1.5B"


def test_display_name_passthrough_from_upstream():
    """上游显式提供 display_name 时透传，不走段表生成。"""
    from deep_proxy.models_list import normalize_model_entry
    out = normalize_model_entry({"id": "weird-name-x", "display_name": "Weird Name X (custom)"})
    assert out["display_name"] == "Weird Name X (custom)"


def test_display_name_empty_string_falls_through():
    """上游 display_name 为空字符串（falsy）时，回退到段表生成而非保留空值。"""
    from deep_proxy.models_list import normalize_model_entry
    out = normalize_model_entry({"id": "deepseek-v4-flash", "display_name": ""})
    assert out["display_name"] == "DeepSeek V4 Flash"


def test_seg_overrides_is_immutable():
    """_SEG_OVERRIDES 用 MappingProxyType 包裹，外部代码无法 mutation。"""
    from deep_proxy.models_list import _SEG_OVERRIDES
    import pytest as _pt
    with _pt.raises(TypeError):
        _SEG_OVERRIDES["foo"] = "Bar"  # type: ignore[index]


def test_anthropic_fields_present_no_capabilities():
    """Anthropic 兼容字段全部输出；不输出杜撰的 capabilities 字段。"""
    from deep_proxy.models_list import normalize_model_entry
    out = normalize_model_entry({"id": "deepseek-v4-flash"})
    assert out["type"] == "model"
    assert out["display_name"] == "DeepSeek V4 Flash"
    assert out["max_input_tokens"] == out["context_length"]
    assert out["max_tokens"] == out["max_completion_tokens"]
    # ISO 8601 UTC 形态：YYYY-MM-DDTHH:MM:SSZ
    assert out["created_at"].endswith("Z") and "T" in out["created_at"]
    # 不输出杜撰的 capabilities（Anthropic 真实 /v1/models 不含此字段）
    assert "capabilities" not in out


def test_upstream_field_precedence_anthropic_first():
    """上游同时给 Anthropic 与 OpenRouter 字段时，Anthropic 原生赢。"""
    from deep_proxy.models_list import normalize_model_entry
    out = normalize_model_entry({
        "id": "test",
        "context_length": 128000,
        "max_input_tokens": 1000000,        # 应胜出
        "max_completion_tokens": 4096,
        "max_tokens": 384000,                # 应胜出
    })
    assert out["context_length"] == 1000000
    assert out["max_input_tokens"] == 1000000
    assert out["max_completion_tokens"] == 384000
    assert out["max_tokens"] == 384000


async def test_anthropic_pagination_wrapper():
    """list_models 响应顶层含 Anthropic 分页字段。"""
    cfg = ProxyConfig(deepseek=DeepSeekConfig(api_key="sk"))
    r = DeepProxyRouter(cfg)
    res = await r.list_models()
    assert "first_id" in res and "last_id" in res
    assert res["has_more"] is False
    assert res["first_id"] == res["data"][0]["id"]
    assert res["last_id"] == res["data"][-1]["id"]
