"""测试 strip_unsupported_params + deepseek_specific_adjustments。

注：经核对官方文档，V4 全面支持 temperature/top_p/.../response_format，
不再按 thinking 模式做特殊剥离。
"""
from __future__ import annotations

import pytest

from deep_proxy.compatibility.error_mapper import strip_unsupported_params


def _filter(model: str, thinking: str | None, **extra) -> dict:
    body = {"model": model, **extra}
    if thinking is not None:
        body["thinking"] = {"type": thinking}
    return strip_unsupported_params(body)


class TestParamFilterMatrix:
    """V4 全部模型 × thinking 状态：保留所有官方支持的参数。"""

    @pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
    @pytest.mark.parametrize("thinking", ["disabled", "enabled"])
    def test_v4_keeps_all_supported_params(self, model: str, thinking: str):
        b = _filter(
            model, thinking,
            temperature=0.5, top_p=0.9,
            presence_penalty=0.1, frequency_penalty=0.2,
            response_format={"type": "json_object"},
            tool_choice="auto",
        )
        assert b["temperature"] == 0.5
        assert b["top_p"] == 0.9
        assert b["presence_penalty"] == 0.1
        assert b["frequency_penalty"] == 0.2
        assert "response_format" in b
        assert b["tool_choice"] == "auto"

    def test_non_v4_keeps_response_format(self):
        b = _filter("gpt-4", None, temperature=0.5, response_format={"type": "json_object"})
        assert b["temperature"] == 0.5
        assert "response_format" in b

    def test_legacy_functions_always_stripped(self):
        b = _filter("deepseek-v4-flash", "disabled", functions=[{"name": "x"}])
        assert "functions" not in b

    def test_user_field_stripped(self):
        b = _filter("deepseek-v4-flash", "disabled", user="end-user-id")
        assert "user" not in b
