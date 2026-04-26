"""测试模型名规范化、is_v4_model、stream_options 清理。"""
from __future__ import annotations

import pytest

from deep_proxy.compatibility.deepseek_fixes import (
    is_v4_model,
    normalize_model_name,
    sanitize_stream_options,
)


class TestIsV4Model:
    def test_v4_explicit(self):
        assert is_v4_model("deepseek-v4-flash") is True
        assert is_v4_model("deepseek-v4-pro") is True

    def test_v4_legacy_aliases(self):
        assert is_v4_model("deepseek-chat") is True
        assert is_v4_model("deepseek-reasoner") is True

    def test_non_v4(self):
        assert is_v4_model("gpt-4") is False
        assert is_v4_model("") is False
        assert is_v4_model("deepseek-coder") is False  # 不在别名里


class TestNormalizeModelName:
    def test_v4_passthrough(self):
        assert normalize_model_name("deepseek-v4-flash") == "deepseek-v4-flash"
        assert normalize_model_name("deepseek-v4-pro") == "deepseek-v4-pro"

    def test_legacy_alias_to_v4(self):
        # 官方文档：reasoner 也是 v4-flash 的别名（思考模式），不是 v4-pro
        assert normalize_model_name("deepseek-chat") == "deepseek-v4-flash"
        assert normalize_model_name("deepseek-reasoner") == "deepseek-v4-flash"

    def test_user_route_overrides_default_alias(self):
        routes = [{"model_name": "deepseek-chat", "provider_model": "custom-chat"}]
        assert normalize_model_name("deepseek-chat", routes) == "custom-chat"

    def test_unknown_passthrough(self):
        assert normalize_model_name("openai/gpt-4") == "openai/gpt-4"

    def test_empty(self):
        assert normalize_model_name("") == ""


class TestSanitizeStreamOptions:
    def test_v4_keeps_include_usage(self):
        b = sanitize_stream_options({
            "model": "deepseek-v4-flash",
            "stream_options": {"include_usage": True},
        })
        assert b["stream_options"] == {"include_usage": True}

    def test_non_v4_keeps_include_usage(self):
        # V4/非 V4 都不再剥离 include_usage（OpenAI 与 DeepSeek 都支持）
        b = sanitize_stream_options({
            "model": "gpt-4",
            "stream_options": {"include_usage": True},
        })
        assert b["stream_options"] == {"include_usage": True}

    def test_empty_stream_options_removed(self):
        b = sanitize_stream_options({
            "model": "deepseek-v4-flash",
            "stream_options": {},
        })
        assert "stream_options" not in b
