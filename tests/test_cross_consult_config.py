"""CrossConsultConfig pydantic 模型测试。"""
from __future__ import annotations

from deep_proxy.cross_consult.config import CrossConsultConfig


def test_cross_consult_config_defaults():
    """默认 enabled=True（plan §1）；未配置 pairs 时 pair_for 仍返回 None，无副作用。"""
    c = CrossConsultConfig()
    assert c.enabled is True
    assert c.tool_name == "cross_consult"
    assert c.pairs == {}
    assert c.pair_for("anything") is None  # 无 pair → 不触发
    assert c.max_calls_per_request == 3
    # §12.11 新字段默认值
    assert c.redirect_enabled is True
    assert c.redirect_persist_turns == 2
    assert c.awareness_enabled is True


def test_cross_consult_config_with_pairs():
    c = CrossConsultConfig(
        enabled=True,
        pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    assert c.enabled is True
    assert c.pairs["deepseek"] == "mimo"
    assert c.pairs["mimo"] == "deepseek"


def test_cross_consult_config_pair_lookup():
    c = CrossConsultConfig(
        enabled=True,
        pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    assert c.pair_for("deepseek") == "mimo"
    assert c.pair_for("mimo") == "deepseek"
    assert c.pair_for("unknown") is None


def test_cross_consult_config_disabled_pair_lookup_returns_none():
    """enabled=False 时 pair_for 一律返回 None（防止意外触发）。"""
    c = CrossConsultConfig(
        enabled=False,
        pairs={"deepseek": "mimo"},
    )
    assert c.pair_for("deepseek") is None


def test_cross_consult_config_has_default_system_prompt():
    c = CrossConsultConfig()
    assert "顾问" in c.consult_system_prompt
    assert "self-contained" in c.consult_system_prompt or "上下文" in c.consult_system_prompt


def test_proxyconfig_cross_consult_enabled_but_inert_without_pairs():
    """默认 enabled=True 但未配置 pairs → pair_for 返回 None，机制不会激活。

    这保证老 config.yaml 不配置 pairs 时升级到新默认安全（不会突然向陌生
    provider 发请求）。
    """
    from deep_proxy.config import ProxyConfig, normalize_legacy_config
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk"},
    }))
    assert cfg.cross_consult.enabled is True
    assert cfg.cross_consult.pair_for("deepseek") is None


def test_proxyconfig_loads_cross_consult_from_yaml():
    from deep_proxy.config import ProxyConfig
    raw = {
        "providers": {
            "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "deepseek/", "flash_model": "a", "pro_model": "b"},
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/", "flash_model": "m", "pro_model": "mp"},
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "y"},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
            "max_calls_per_request": 2,
        },
    }
    cfg = ProxyConfig.model_validate(raw)
    assert cfg.cross_consult.enabled is True
    assert cfg.cross_consult.pair_for("deepseek") == "mimo"
    assert cfg.cross_consult.max_calls_per_request == 2
