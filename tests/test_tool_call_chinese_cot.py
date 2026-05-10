"""Tests for tool_call_chinese_cot —— tools 场景中文 CoT 双通路锚定。

设计依据: docs/superpowers/specs/2026-05-10-tool-call-chinese-cot-design.md
研究依据: fkyah3/experiment-console
"""

from __future__ import annotations

import pytest

from deep_proxy.optimization import apply_cheap_optimizations
from deep_proxy.optimization.skills_general import (
    _SKILL_AVOID_AI_TICS,
    _SKILL_COT_RESET,
    _SKILL_INSTRUCTION_PRIORITY,
    _SKILL_REASON_GENUINELY,
    _SKILL_SHOW_MATH_STEPS,
)
from deep_proxy.optimization.think_steering import (
    INNER_OS_MARKER,
    inject_inner_os_marker,
)
from deep_proxy.optimization.tool_call_chinese_cot import (
    TOOL_CALL_CN_COT_SKILL,
    TOOL_CALL_CN_COT_USER_MARKER,
    _MARKER_SIGNATURE,
    has_tool_call_cn_cot_marker,
    inject_user_marker,
)


# ──────────────────────────────────────────────────────────────────────────
# 单元：模块本身（marker 检测 / 注入）
# ──────────────────────────────────────────────────────────────────────────


class TestHasMarker:
    def test_no_marker(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        assert has_tool_call_cn_cot_marker(messages) is False

    def test_marker_present_in_first_user(self):
        messages = [
            {"role": "user", "content": f"hi.\n\n【{_MARKER_SIGNATURE}】"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        assert has_tool_call_cn_cot_marker(messages) is True

    def test_marker_only_in_system_does_not_count(self):
        """system 通路自己控（dedup），这里只关心 user 通路。"""
        messages = [
            {"role": "system", "content": _MARKER_SIGNATURE},
            {"role": "user", "content": "hi"},
        ]
        assert has_tool_call_cn_cot_marker(messages) is False

    def test_empty(self):
        assert has_tool_call_cn_cot_marker([]) is False


class TestInjectUserMarker:
    def test_dual_inject_first_and_last(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "middle"},
            {"role": "assistant", "content": "reply2"},
            {"role": "user", "content": "last"},
        ]
        assert inject_user_marker(messages) is True
        assert messages[1]["content"].endswith(TOOL_CALL_CN_COT_USER_MARKER)
        assert messages[5]["content"].endswith(TOOL_CALL_CN_COT_USER_MARKER)
        # 中间 user 不动
        assert messages[3]["content"] == "middle"

    def test_single_turn_inject_once(self):
        messages = [{"role": "user", "content": "only"}]
        assert inject_user_marker(messages) is True
        assert messages[0]["content"] == "only" + TOOL_CALL_CN_COT_USER_MARKER

    def test_idempotent(self):
        messages = [
            {"role": "user", "content": "hi" + TOOL_CALL_CN_COT_USER_MARKER},
        ]
        before = messages[0]["content"]
        assert inject_user_marker(messages) is False
        assert messages[0]["content"] == before

    def test_no_user_returns_false(self):
        messages = [{"role": "system", "content": "sys"}]
        assert inject_user_marker(messages) is False

    def test_skips_non_string_content(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        assert inject_user_marker(messages) is False


# ──────────────────────────────────────────────────────────────────────────
# 集成：与 apply_cheap_optimizations 的 has_tools 分流
# ──────────────────────────────────────────────────────────────────────────


def _tools_body(messages, **extra):
    """构造一个带 tools 字段的 body —— 触发 has_tools 分流。"""
    return {
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List a directory.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        **extra,
    }


class TestToolsBranchInjection:
    async def test_injects_system_prefix_and_user_dual(self):
        """tools 场景：system 前缀 + user 首/末双注入都生效。"""
        body = _tools_body(
            [
                {"role": "system", "content": "你是一个 AI 工程助手。"},
                {"role": "user", "content": "请分析项目。"},
                {"role": "assistant", "content": "好的。"},
                {"role": "user", "content": "继续。"},
            ]
        )
        await apply_cheap_optimizations(body)

        # system 前缀含 4 条 skills + 用户原 system
        sys_content = body["messages"][0]["content"]
        assert _SKILL_INSTRUCTION_PRIORITY in sys_content
        assert _SKILL_REASON_GENUINELY in sys_content
        assert _SKILL_COT_RESET in sys_content
        assert TOOL_CALL_CN_COT_SKILL in sys_content
        assert "你是一个 AI 工程助手。" in sys_content
        # inject_date 也已追加
        assert "今天是" in sys_content

        # user 首/末双注入
        assert body["messages"][1]["content"].endswith(TOOL_CALL_CN_COT_USER_MARKER)
        assert body["messages"][3]["content"].endswith(TOOL_CALL_CN_COT_USER_MARKER)

    async def test_no_excluded_skills_leak_into_tools_path(self):
        """tools 路径下，A/B/C 组其他 skills 不应混入 system 前缀。"""
        body = _tools_body(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ]
        )
        # 即使把所有 kwargs 都开到 True（默认就是），tools 路径仍只跑最小化 pipeline
        await apply_cheap_optimizations(body)

        sys_content = body["messages"][0]["content"]
        # B 组的 show_math_steps 不应出现
        assert _SKILL_SHOW_MATH_STEPS not in sys_content
        # A 组里被排除的 avoid_negative_style (avoid AI tics) 不应出现
        assert _SKILL_AVOID_AI_TICS not in sys_content

    async def test_disabled_via_kwarg(self):
        """tool_call_chinese_cot=False 时 tools 路径不注入任何东西。"""
        original_sys = "你是工程助手。"
        original_user = "分析一下。"
        body = _tools_body(
            [
                {"role": "system", "content": original_sys},
                {"role": "user", "content": original_user},
            ]
        )
        await apply_cheap_optimizations(body, tool_call_chinese_cot=False)

        # system / user 内容完全保持原样
        assert body["messages"][0]["content"] == original_sys
        assert body["messages"][1]["content"] == original_user

    async def test_thinking_disabled_skips(self):
        """thinking.type=disabled 时不注入（无 reasoning 也就没漂移）。"""
        original_sys = "sys"
        original_user = "q"
        body = _tools_body(
            [
                {"role": "system", "content": original_sys},
                {"role": "user", "content": original_user},
            ],
            thinking={"type": "disabled"},
        )
        await apply_cheap_optimizations(body)

        assert body["messages"][0]["content"] == original_sys
        assert body["messages"][1]["content"] == original_user

    async def test_thinking_enabled_injects(self):
        """thinking.type=enabled 时正常注入（默认行为）。"""
        body = _tools_body(
            [
                {"role": "user", "content": "q"},
            ],
            thinking={"type": "enabled"},
        )
        await apply_cheap_optimizations(body)

        # user 末尾有 marker
        assert body["messages"][-1]["content"].endswith(TOOL_CALL_CN_COT_USER_MARKER)

    async def test_thinking_field_absent_treated_as_enabled(self):
        """thinking 字段缺失时按 enabled 处理（V4 服务端默认）。"""
        body = _tools_body([{"role": "user", "content": "q"}])
        # 没设 thinking
        await apply_cheap_optimizations(body)
        # 应当注入
        assert body["messages"][-1]["content"].endswith(TOOL_CALL_CN_COT_USER_MARKER)


class TestNoToolsUnaffected:
    """无 tools 时本 skill 不触发，原 pipeline 完整运行。"""

    async def test_no_tools_does_not_inject_user_marker(self):
        body = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ]
        }
        await apply_cheap_optimizations(body)
        # user 不应被追加 tool_call_chinese_cot 的 user marker
        # （但可能被原 pipeline 的其他 skills 修改 system）
        for msg in body["messages"]:
            if msg.get("role") == "user":
                assert TOOL_CALL_CN_COT_USER_MARKER not in msg["content"]


class TestCoexistsWithThinkSteering:
    """与 think_steering 的 inner-OS marker 共存，互不干扰。"""

    async def test_both_markers_in_user_content(self):
        body = _tools_body(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ]
        )
        # 先打 inner_os_marker（模拟 router.py 流程：creative mode 后追加）
        inject_inner_os_marker(body["messages"])
        # 再走 apply_cheap_optimizations（tools 分流注入 tool_call_chinese_cot user marker）
        await apply_cheap_optimizations(body)

        user_content = body["messages"][-1]["content"]
        # 两个 marker 都在
        assert "【角色沉浸要求】" in user_content
        assert _MARKER_SIGNATURE in user_content
        # signature 不互相吞噬
        assert user_content.count(_MARKER_SIGNATURE) >= 1


class TestIdempotentOnRetry:
    """同一 body 二次穿过不重复注入（_deepproxy_optimized 防护）。"""

    async def test_double_apply_is_noop(self):
        body = _tools_body([{"role": "user", "content": "q"}])
        await apply_cheap_optimizations(body)
        snapshot = [dict(m) for m in body["messages"]]
        # 二次穿过
        await apply_cheap_optimizations(body)
        assert body["messages"] == snapshot
