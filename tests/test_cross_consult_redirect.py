"""标签触发的整轮 provider 重定向测试（plan §12.11）。"""
from __future__ import annotations

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.cross_consult.redirect import _scan_and_strip, resolve_redirect
from deep_proxy.cross_consult.redirect_tracker import RedirectTracker


REDIRECT_TAG = "[本轮对话使用不同家族的大语言模型]"


@pytest.fixture
def cfg_dual_with_redirect() -> ProxyConfig:
    """双 provider + 默认开启 cross_consult + pairs + redirect。"""
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek", "api_base": "x", "api_key": "y",
                "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash", "pro_model": "deepseek-v4-pro",
            },
            "mimo": {
                "name": "mimo", "api_base": "x", "api_key": "y",
                "litellm_prefix": "openai/",
                "flash_model": "mimo-v2.5", "pro_model": "mimo-v2.5-pro",
            },
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "y"},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
            "redirect_persist_turns": 2,
        },
    })


# ---------------------------------------------------------------------------
# _scan_and_strip 单元
# ---------------------------------------------------------------------------


def test_scan_and_strip_hits_last_user_only_triggers():
    """最后一条 user 含标签 → 命中（返回 True）。"""
    import re
    pat = re.compile(r"\[本轮对话使用不同家族的大语言模型\]")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": f"new question {REDIRECT_TAG}"},
    ]
    assert _scan_and_strip(messages, pat) is True
    # 最后一条 user 标签已剥离
    assert REDIRECT_TAG not in messages[-1]["content"]
    assert "new question" in messages[-1]["content"]


def test_scan_and_strip_historical_tag_strips_but_not_triggers():
    """历史 user 含标签、最后 user 不含 → 剥离但不触发（返回 False）。"""
    import re
    pat = re.compile(r"\[本轮对话使用不同家族的大语言模型\]")
    messages = [
        {"role": "user", "content": f"old req {REDIRECT_TAG}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "clean new req"},
    ]
    assert _scan_and_strip(messages, pat) is False
    # 但历史标签也被剥离了（防止上游模型在 history 看到 meta-instruction）
    assert REDIRECT_TAG not in messages[0]["content"]
    assert messages[-1]["content"] == "clean new req"


def test_scan_and_strip_empty_after_strip_keeps_space():
    """剥离后 content 为空 → 回填单空格避免上游 422。"""
    import re
    pat = re.compile(r"\[本轮对话使用不同家族的大语言模型\]")
    messages = [{"role": "user", "content": REDIRECT_TAG}]
    assert _scan_and_strip(messages, pat) is True
    assert messages[0]["content"] == " "


def test_scan_and_strip_multimodal_content():
    """list-of-blocks content（多模态）按 block 剥离。"""
    import re
    pat = re.compile(r"\[本轮对话使用不同家族的大语言模型\]")
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"hi {REDIRECT_TAG}"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    assert _scan_and_strip(messages, pat) is True
    assert REDIRECT_TAG not in messages[0]["content"][0]["text"]
    # 非 text block 不动
    assert messages[0]["content"][1]["type"] == "image_url"


def test_scan_and_strip_no_user_messages_returns_false():
    import re
    pat = re.compile(r"\[本轮对话使用不同家族的大语言模型\]")
    messages = [{"role": "system", "content": REDIRECT_TAG}]
    assert _scan_and_strip(messages, pat) is False


# ---------------------------------------------------------------------------
# resolve_redirect 集成
# ---------------------------------------------------------------------------


def test_resolve_redirect_tag_hit_returns_target(cfg_dual_with_redirect):
    """标签命中 → 返回异家族 provider + tracker 计数被设置。"""
    cfg = cfg_dual_with_redirect
    tracker = RedirectTracker()
    src = cfg.providers["deepseek"]
    body = {"messages": [{"role": "user", "content": f"writing task {REDIRECT_TAG}"}]}
    target = resolve_redirect(
        body, source_provider=src, config=cfg, tracker=tracker,
    )
    assert target is not None
    assert target.name == "mimo"
    # 标签已剥离
    assert REDIRECT_TAG not in body["messages"][-1]["content"]
    # tracker 中应有计数（persist_turns=2 + 1 本轮 = 3）
    assert tracker.remaining(
        body["messages"], source_provider_name="deepseek"
    ) == 3


def test_resolve_redirect_no_tag_no_window_returns_none(cfg_dual_with_redirect):
    """无标签、无 persist 窗口 → None。"""
    cfg = cfg_dual_with_redirect
    tracker = RedirectTracker()
    src = cfg.providers["deepseek"]
    body = {"messages": [{"role": "user", "content": "plain"}]}
    target = resolve_redirect(
        body, source_provider=src, config=cfg, tracker=tracker,
    )
    assert target is None


def test_resolve_redirect_persist_window_continues_without_tag(cfg_dual_with_redirect):
    """触发后下一轮无标签仍走异家族（在 persist 窗口内）。"""
    cfg = cfg_dual_with_redirect  # persist_turns=2
    tracker = RedirectTracker()
    src = cfg.providers["deepseek"]

    # 轮 1：带标签触发
    body1 = {"messages": [
        {"role": "user", "content": f"start {REDIRECT_TAG}"},
    ]}
    t1 = resolve_redirect(body1, source_provider=src, config=cfg, tracker=tracker)
    assert t1.name == "mimo"

    # 轮 2：同对话，新 user 消息，无标签
    body2 = {"messages": [
        {"role": "user", "content": "start"},  # 已剥离的版本
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "follow up no tag"},
    ]}
    t2 = resolve_redirect(body2, source_provider=src, config=cfg, tracker=tracker)
    assert t2 is not None and t2.name == "mimo"

    # 轮 3：再下一轮，仍在窗口（persist_turns=2 → 含本次共 3 轮）
    body3 = {"messages": [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "follow up no tag"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "third turn"},
    ]}
    t3 = resolve_redirect(body3, source_provider=src, config=cfg, tracker=tracker)
    assert t3 is not None and t3.name == "mimo"

    # 轮 4：窗口耗尽，回到源 provider
    body4 = {"messages": [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "follow up no tag"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "third turn"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "fourth turn"},
    ]}
    t4 = resolve_redirect(body4, source_provider=src, config=cfg, tracker=tracker)
    assert t4 is None


def test_resolve_redirect_disabled_when_redirect_enabled_false():
    """redirect_enabled=False → 即使有标签也不重定向。"""
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
            "redirect_enabled": False,
        },
    })
    tracker = RedirectTracker()
    body = {"messages": [{"role": "user", "content": f"x {REDIRECT_TAG}"}]}
    target = resolve_redirect(
        body, source_provider=cfg.providers["deepseek"],
        config=cfg, tracker=tracker,
    )
    assert target is None
    # 但标签也不剥（resolve_redirect 直接 return None）— 仅当 redirect 启用时才扫描
    assert REDIRECT_TAG in body["messages"][-1]["content"]


def test_resolve_redirect_fail_open_when_pair_missing(cfg_dual_with_redirect, caplog):
    """pairs 配置存在但 target_provider 不在 providers map 中 → 不重定向 + warn log。"""
    import logging
    cfg = cfg_dual_with_redirect.model_copy(deep=True)
    # 人为构造一个指向不存在 provider 的 pair
    cfg.cross_consult.pairs = {"deepseek": "ghost"}
    tracker = RedirectTracker()
    body = {"messages": [{"role": "user", "content": f"x {REDIRECT_TAG}"}]}
    with caplog.at_level(logging.WARNING, logger="deep_proxy.cross_consult.redirect"):
        target = resolve_redirect(
            body, source_provider=cfg.providers["deepseek"],
            config=cfg, tracker=tracker,
        )
    assert target is None
    assert any("fail_open" in rec.message for rec in caplog.records)


def test_resolve_redirect_no_pair_for_source_returns_none(cfg_dual_with_redirect):
    """source provider 不在 pairs 中 → 不重定向（早期返回，不扫描）。"""
    cfg = cfg_dual_with_redirect.model_copy(deep=True)
    cfg.cross_consult.pairs = {"deepseek": "mimo"}  # 无 mimo→deepseek
    tracker = RedirectTracker()
    body = {"messages": [{"role": "user", "content": f"x {REDIRECT_TAG}"}]}
    target = resolve_redirect(
        body, source_provider=cfg.providers["mimo"],
        config=cfg, tracker=tracker,
    )
    assert target is None


def test_resolve_redirect_skips_recursion_sentinel(cfg_dual_with_redirect):
    """body 带 _deepproxy_cross_consult_internal sentinel → 不重定向。"""
    tracker = RedirectTracker()
    cfg = cfg_dual_with_redirect
    body = {
        "messages": [{"role": "user", "content": f"x {REDIRECT_TAG}"}],
        "_deepproxy_cross_consult_internal": True,
    }
    target = resolve_redirect(
        body, source_provider=cfg.providers["deepseek"],
        config=cfg, tracker=tracker,
    )
    assert target is None
