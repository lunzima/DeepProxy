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
