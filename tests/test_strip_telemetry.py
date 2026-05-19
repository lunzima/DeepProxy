"""测试 telemetry header 行剥离。

覆盖：
- 单行 / 多行剥离
- 大小写不敏感
- 任意 x-anthropic-* 子名匹配
- 不误删非行首位置的同名字符串
- str / list[dict] 两种 system content 形态
- 首条 user 消息也覆盖
- None / [] / 缺失字段的 no-op 容错
"""
from __future__ import annotations

from deep_proxy.optimization.strip_telemetry import (
    _TELEMETRY_HEADER_RE,
    strip_telemetry_from_text,
    strip_telemetry_from_messages,
)


class TestStripText:
    def test_single_header_line_removed(self):
        text = (
            "x-anthropic-billing-header: cc_version=2.1.42.abc; cc_entrypoint=claude-code; cch=0;\n"
            "You are Claude Code."
        )
        out = strip_telemetry_from_text(text)
        assert "x-anthropic-billing-header" not in out
        assert "You are Claude Code." in out

    def test_multiple_x_anthropic_lines_removed(self):
        text = (
            "x-anthropic-foo: a\n"
            "x-anthropic-bar: b\n"
            "real content"
        )
        out = strip_telemetry_from_text(text)
        assert "x-anthropic-foo" not in out
        assert "x-anthropic-bar" not in out
        assert "real content" in out

    def test_case_insensitive(self):
        text = "X-Anthropic-Billing-Header: foo\nreal"
        out = strip_telemetry_from_text(text)
        assert "Billing-Header" not in out
        assert "real" in out

    def test_inline_mention_not_removed(self):
        # 仅匹配行首；正文里引用 "x-anthropic-foo: ..." 不应被吃掉
        text = "Quote: 'x-anthropic-foo: bar' as example"
        out = strip_telemetry_from_text(text)
        assert "x-anthropic-foo: bar" in out

    def test_empty_input(self):
        assert strip_telemetry_from_text("") == ""
        assert strip_telemetry_from_text(None) is None  # type: ignore[arg-type]

    def test_no_match_returns_unchanged(self):
        text = "Plain system prompt with no telemetry."
        assert strip_telemetry_from_text(text) == text


class TestStripMessages:
    def test_system_str_content_stripped(self):
        messages = [
            {"role": "system", "content": "x-anthropic-billing-header: foo\nYou are helpful."},
            {"role": "user", "content": "hi"},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-billing-header" not in messages[0]["content"]
        assert "You are helpful." in messages[0]["content"]

    def test_system_list_content_stripped(self):
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "x-anthropic-foo: bar\nrules"},
                {"type": "text", "text": "more rules"},
            ]},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-foo" not in messages[0]["content"][0]["text"]
        assert messages[0]["content"][1]["text"] == "more rules"

    def test_first_user_message_also_stripped(self):
        messages = [
            {"role": "user", "content": "x-anthropic-billing-header: foo\nactual question"},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-billing-header" not in messages[0]["content"]
        assert "actual question" in messages[0]["content"]

    def test_later_user_message_untouched(self):
        # 只清理首条 user（CC 注入位置）；后续 user 消息可能合法引用该字符串
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "x-anthropic-foo: should NOT be touched here"},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-foo" in messages[3]["content"]

    def test_empty_messages_noop(self):
        strip_telemetry_from_messages([])  # 不抛异常
        strip_telemetry_from_messages(None)  # type: ignore[arg-type]

    def test_non_dict_message_skipped(self):
        # 容错：遇到非 dict 项不抛
        messages = ["bogus", {"role": "system", "content": "x-anthropic-foo: bar\nok"}]
        strip_telemetry_from_messages(messages)  # type: ignore[arg-type]
        assert "x-anthropic-foo" not in messages[1]["content"]

    def test_non_str_content_left_alone(self):
        # content 既非 str 也非 list[dict]：静默跳过，不抛
        messages = [{"role": "system", "content": 12345}]
        strip_telemetry_from_messages(messages)  # type: ignore[list-item]
        assert messages[0]["content"] == 12345


class TestRegex:
    def test_regex_pattern_anchored_to_line_start(self):
        # 文档化共享正则的行为
        import re
        assert _TELEMETRY_HEADER_RE.flags & re.MULTILINE
        assert _TELEMETRY_HEADER_RE.flags & re.IGNORECASE
