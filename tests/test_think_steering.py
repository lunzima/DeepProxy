"""Tests for think_steering.py — V4 <think> 角色沉浸引导模块。"""

from __future__ import annotations

from deep_proxy.optimization.think_steering import (
    INNER_OS_MARKER,
    _MARKER_SIGNATURE,
    has_inner_os_marker,
    inject_inner_os_marker,
)


class TestHasInnerOsMarker:
    def test_no_marker(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello."},
        ]
        assert has_inner_os_marker(messages) is False

    def test_marker_present(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Hello.\n\n{_MARKER_SIGNATURE} more stuff"},
        ]
        assert has_inner_os_marker(messages) is True

    def test_marker_in_older_user_message(self):
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": f"First turn.\n\n{_MARKER_SIGNATURE}"},
            {"role": "assistant", "content": "Reply 1."},
            {"role": "user", "content": "Second turn."},
        ]
        assert has_inner_os_marker(messages) is True

    def test_marker_incomplete_signature(self):
        """部分匹配不作数，必须完整特征字符串才算。"""
        messages = [
            {"role": "user", "content": "角色沉浸要求"},
        ]
        assert has_inner_os_marker(messages) is False

    def test_empty_messages(self):
        assert has_inner_os_marker([]) is False


class TestInjectInnerOsMarker:
    def test_injects_to_last_user(self):
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "user", "content": "Second."},
        ]
        result = inject_inner_os_marker(messages)
        assert result is True
        assert messages[3]["content"].endswith(INNER_OS_MARKER)
        # 之前的消息不受影响
        assert messages[1]["content"] == "First."

    def test_idempotent_already_has_marker(self):
        messages = [
            {"role": "user", "content": f"Hello.\n\n{_MARKER_SIGNATURE}"},
        ]
        original_len = len(messages[0]["content"])
        result = inject_inner_os_marker(messages)
        assert result is False  # 没注入
        assert len(messages[0]["content"]) == original_len  # 内容不变

    def test_no_user_message(self):
        messages = [
            {"role": "system", "content": "System only."},
        ]
        result = inject_inner_os_marker(messages)
        assert result is False

    def test_empty_user_content(self):
        messages = [
            {"role": "user", "content": ""},
        ]
        result = inject_inner_os_marker(messages)
        assert result is False

    def test_empty_messages_list(self):
        result = inject_inner_os_marker([])
        assert result is False
