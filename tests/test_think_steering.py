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
    def test_injects_to_first_and_last(self):
        """多轮对话：第一条和最后一条 user 都注入，中间的不动。"""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "First turn."},
            {"role": "assistant", "content": "Reply 1."},
            {"role": "user", "content": "Middle turn."},
            {"role": "assistant", "content": "Reply 2."},
            {"role": "user", "content": "Last turn."},
        ]
        result = inject_inner_os_marker(messages)
        assert result is True
        # 第一条 user 末尾有 marker
        assert messages[1]["content"].endswith(INNER_OS_MARKER)
        # 最后一条 user 末尾也有 marker
        assert messages[5]["content"].endswith(INNER_OS_MARKER)
        # 中间的 user 不受影响
        assert messages[3]["content"] == "Middle turn."

    def test_single_turn_no_duplicate(self):
        """单轮对话：第一条==最后一条，只注入一次。"""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Only turn."},
        ]
        result = inject_inner_os_marker(messages)
        assert result is True
        # 内容恰好以 marker 结尾（只追加了一次）
        expected = "Only turn." + INNER_OS_MARKER
        assert messages[1]["content"] == expected

    def test_idempotent_already_has_marker(self):
        """已有 marker 在任何 user 消息中 → 整体跳过。"""
        messages = [
            {"role": "user", "content": f"Hello.\n\n{_MARKER_SIGNATURE}"},
        ]
        original_len = len(messages[0]["content"])
        result = inject_inner_os_marker(messages)
        assert result is False  # 没注入
        assert len(messages[0]["content"]) == original_len  # 内容不变

    def test_idempotent_marker_in_middle_message(self):
        """marker 在中间 user 消息中 → 整体跳过（不追加到最后）。"""
        messages = [
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "user", "content": f"Second.\n\n{_MARKER_SIGNATURE}"},
            {"role": "assistant", "content": "Reply 2."},
            {"role": "user", "content": "Third."},
        ]
        # 记录注入前的最后一条内容
        original_last = messages[4]["content"]
        result = inject_inner_os_marker(messages)
        assert result is False
        # 所有内容不变
        assert messages[4]["content"] == original_last
        assert messages[0]["content"] == "First."

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
