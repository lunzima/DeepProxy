"""双家族 awareness 注入测试（plan §2）。"""
from __future__ import annotations

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.cross_consult.awareness import build_awareness_prompt
from deep_proxy.cross_consult.interceptor import inject_into_request
from deep_proxy.cross_consult.config import CrossConsultConfig


# ---------------------------------------------------------------------------
# build_awareness_prompt
# ---------------------------------------------------------------------------


def test_awareness_names_both_providers():
    text = build_awareness_prompt(
        source_provider_name="deepseek", target_provider_name="mimo",
    )
    assert "deepseek" in text
    assert "mimo" in text


def test_awareness_mentions_three_choices_and_tag_literal():
    text = build_awareness_prompt(
        source_provider_name="deepseek", target_provider_name="mimo",
        tool_name="cross_consult", max_calls=3,
    )
    assert "[本轮对话使用不同家族的大语言模型]" in text
    assert "cross_consult" in text
    assert "保持" in text  # 选项 1
    assert "单点" in text or "第二视角" in text  # 选项 2
    assert "多轮" in text or "整轮" in text  # 选项 3
    # max_calls 是允许暴露的少数数值
    assert "3" in text


def test_awareness_hides_internal_implementation():
    """禁止泄露：BERT/router/heuristic/threshold/score/persist_turns/具体阈值数值，以及 flash/pro 档位字眼。"""
    text = build_awareness_prompt(
        source_provider_name="deepseek", target_provider_name="mimo",
    )
    # 这些字符串绝不应出现在 awareness 文案中（即便子串）
    forbidden_substrings = [
        "BERT", "bert", "router_threshold", "heuristic", "Heuristic",
        "threshold", "Threshold", "persist_turns",
        "0.60", "0.65", "7.5", "8.0",
        "flash", "Flash",
    ]
    for kw in forbidden_substrings:
        assert kw not in text, f"awareness 不应暴露 {kw!r}"

    # "pro" 子串易与品牌名 "DeepProxy" 冲突——按 token 边界检查更准确
    # 这些组合都暗示档位选择，必须不出现
    forbidden_phrases = ["pro 模型", "Pro 模型", "pro model", "pro_model"]
    for ph in forbidden_phrases:
        assert ph not in text, f"awareness 不应暴露档位字眼 {ph!r}"


def test_awareness_mentions_window_returns_to_source():
    """文案需告诉 agent：窗口耗尽后会自然回到源家族，需要继续要重新插标签。"""
    text = build_awareness_prompt(
        source_provider_name="deepseek", target_provider_name="mimo",
    )
    assert "回到" in text or "回归" in text or "返回" in text
    assert "重新" in text or "再次" in text


# ---------------------------------------------------------------------------
# inject_into_request 集成
# ---------------------------------------------------------------------------


def test_inject_includes_awareness_when_enabled():
    cc = CrossConsultConfig(
        enabled=True,
        pairs={"deepseek": "mimo", "mimo": "deepseek"},
        awareness_enabled=True,
    )
    body = {"messages": [{"role": "system", "content": "user sys"},
                         {"role": "user", "content": "hi"}]}
    assert inject_into_request(body, source_provider_name="deepseek", cc_config=cc) is True
    sys = body["messages"][0]["content"]
    assert "双家族披露" in sys
    assert "mimo" in sys
    # tool addendum 也要在
    assert "cross_consult" in sys


def test_inject_skips_awareness_when_awareness_disabled():
    cc = CrossConsultConfig(
        enabled=True,
        pairs={"deepseek": "mimo"},
        awareness_enabled=False,
    )
    body = {"messages": [{"role": "system", "content": "user sys"},
                         {"role": "user", "content": "hi"}]}
    inject_into_request(body, source_provider_name="deepseek", cc_config=cc)
    sys = body["messages"][0]["content"]
    assert "双家族披露" not in sys
    # tool addendum 仍在
    assert "cross_consult" in sys


def test_inject_creates_system_when_missing():
    """无 system 消息 → 创建新 system 包含 awareness + tool addendum。"""
    cc = CrossConsultConfig(
        enabled=True,
        pairs={"deepseek": "mimo"},
        awareness_enabled=True,
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_into_request(body, source_provider_name="deepseek", cc_config=cc)
    assert body["messages"][0]["role"] == "system"
    sys = body["messages"][0]["content"]
    assert "双家族披露" in sys
    assert "cross_consult" in sys


def test_inject_awareness_precedes_tool_addendum_in_system():
    """awareness 段在 tool addendum 之前（plan §2 注入位置）。"""
    cc = CrossConsultConfig(
        enabled=True,
        pairs={"deepseek": "mimo"},
        awareness_enabled=True,
    )
    body = {"messages": [{"role": "system", "content": "USER_SYS"},
                         {"role": "user", "content": "hi"}]}
    inject_into_request(body, source_provider_name="deepseek", cc_config=cc)
    sys = body["messages"][0]["content"]
    awareness_pos = sys.find("双家族披露")
    # tool addendum 的特征短语
    tool_pos = sys.find("你可调用工具")
    assert awareness_pos >= 0
    assert tool_pos >= 0
    assert awareness_pos < tool_pos, (
        f"awareness 应在 tool addendum 之前；awareness@{awareness_pos}, tool@{tool_pos}"
    )
    # USER_SYS 仍在最前
    assert sys.find("USER_SYS") < awareness_pos


async def test_prepare_request_injects_awareness_end_to_end():
    """端到端：prepare_request 走完管道后 system 中包含 awareness。"""
    from deep_proxy.router import DeepProxyRouter

    cfg = ProxyConfig.model_validate({
        "providers": {
            "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "deepseek/",
                         "flash_model": "deepseek-v4-flash", "pro_model": "deepseek-v4-pro"},
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/",
                     "flash_model": "mimo-v2.5", "pro_model": "mimo-v2.5-pro"},
        },
        "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
        "deepseek": {"api_key": "y"},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
            "awareness_enabled": True,
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
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert any("双家族披露" in m.get("content", "") for m in sys_msgs)
    assert any("mimo" in m.get("content", "") for m in sys_msgs)
