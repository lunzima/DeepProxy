"""MiMo 路径下的协议差异适配测试。"""
from __future__ import annotations

from deep_proxy.compatibility.mimo_fixes import (
    inject_top_level_reasoning_effort,
)


def test_inject_top_level_reasoning_effort_no_thinking_no_existing():
    body = {"model": "mimo-v2.5", "messages": []}
    inject_top_level_reasoning_effort(body, value="high")
    assert body["reasoning_effort"] == "high"
    assert "thinking" not in body


def test_inject_skips_when_thinking_disabled():
    body = {"model": "mimo-v2.5", "messages": [], "thinking": {"type": "disabled"}}
    inject_top_level_reasoning_effort(body, value="high")
    # 显式 disabled 时不注入
    assert "reasoning_effort" not in body


def test_inject_respects_existing_reasoning_effort():
    body = {"model": "mimo-v2.5", "messages": [], "reasoning_effort": "low"}
    inject_top_level_reasoning_effort(body, value="high")
    # 客户端已显式设置时不覆盖
    assert body["reasoning_effort"] == "low"
