"""测试 StyleGuard — 扫描 + 反馈 + 重发循环。"""

from __future__ import annotations

import pytest

from deep_proxy.optimization.style_guard import (
    RULES,
    StyleRule,
    _extract_sentence,
    _has_narrative_anchors,
    apply_style_guard_loop,
    build_feedback_message,
    scan_tool_call_violations,
    scan_violations,
)


class TestExtractSentence:
    def test_extracts_full_sentence_with_period(self):
        text = "他坐在椅子上没有动。"
        result = _extract_sentence(text, 6, 9)
        assert result == "他坐在椅子上没有动。"

    def test_extracts_sentence_with_newline_boundary(self):
        text = "推了推\n另一个句子"
        result = _extract_sentence(text, 0, 2)
        assert result == "推了推"

    def test_extracts_sentence_with_exclamation(self):
        text = "目光扫到窗外！风吹了进来。"
        result = _extract_sentence(text, 0, 3)
        assert result == "目光扫到窗外！"


class TestScanViolations:
    def test_a3_one_cut_shadow(self):
        text = "月光平移了一截，照在天花板上。"
        hits = scan_violations(text)
        rule_ids = {h["rule_id"] for h in hits}
        assert "a3" in rule_ids

    def test_a5_half_cun(self):
        text = "他把卷宗推了半寸到夏海东面前。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "a5" for h in hits)

    def test_b1_sound_swallowed(self):
        text = "他的声音被墙壁吞掉了。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "b1" for h in hits)

    def test_d2_cant_tell(self):
        text = "他也说不清楚为什么会这样做。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "d2" for h in hits)

    def test_e1_personification(self):
        text = "建筑提醒他这里不是私人空间。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "e1" for h in hits)

    def test_f1_pov_scan(self):
        text = "他的目光扫到桌角的文件上。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "f1" for h in hits)

    def test_fr1_finger_frame(self):
        text = "两个拇指上下轻轻摩挲了一下。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "fr1" for h in hits)

    def test_fr2_joint_sequence(self):
        text = "从肩膀到颈椎都绷着，下颌压了一下。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "fr2" for h in hits)

    def test_fr4_eye_micro(self):
        text = "瞳孔收缩了一下，聚焦的角度变了。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "fr4" for h in hits)

    def test_fr5_negation(self):
        text = "他站在那里，没有动。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "fr5" for h in hits)

    def test_dash(self):
        text = "他走向办公室——里面已经坐了人。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "dash" for h in hits)

    def test_attr_a_causal(self):
        text = "手在发抖，不是因为害怕，而是因为愤怒。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "attr_a" for h in hits)

    def test_attr_a_variant_without_er(self):
        text = "这不是害怕，是愤怒。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "attr_a" for h in hits)

    def test_clic_stale_metaphor(self):
        text = "像投入湖面的石子，引起层层涟漪。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "clic" for h in hits)

    def test_emo_label(self):
        text = "他感到一阵深深的孤独。"
        hits = scan_violations(text)
        assert any(h["rule_id"] == "emo" for h in hits)

    def test_clean_text_no_violations(self):
        text = "他点了点头，把一个文件夹推到夏海东面前。夏海东接过文件夹，翻开看了一眼。"
        hits = scan_violations(text)
        assert hits == []

    def test_dedup_same_rule_same_sentence(self):
        """同一规则在同一句子中多次命中应去重为一次。"""
        text = "目光扫到桌上，余光扫到墙角。"
        hits = scan_violations(text)
        f1_hits = [h for h in hits if h["rule_id"] == "f1"]
        assert len(f1_hits) <= 1

    def test_identical_sentences_distinct_positions_not_collapsed(self):
        """同形句子出现在不同位置应各报一次（span 去重，非 text.find 首次出现）。"""
        text = "他没有动。中间隔一句拉开距离填充足够字数避免误合并。他没有动。"
        fr5 = [h for h in scan_violations(text) if h["rule_id"] == "fr5"]
        assert len(fr5) == 2


class TestBuildFeedbackMessage:
    def test_contains_pattern_name_and_example_fix(self):
        violations = [{
            "rule_id": "fr5",
            "pattern_name": "否定式描写（以缺席动作替代在场）",
            "sentence": "他站在那里，没有动。",
            "example_fix": "写出此刻的在场动作",
        }]
        msg = build_feedback_message(violations)
        assert "禁用的写作模式" in msg
        assert "否定式描写" in msg
        assert "没有动" in msg
        assert "在场动作" in msg
        assert "重新提交" in msg

    def test_multiple_violations_numbered(self):
        violations = [
            {"rule_id": "a5", "pattern_name": "半寸/几息", "sentence": "推了半寸。", "example_fix": "动词重叠式"},
            {"rule_id": "f1", "pattern_name": "目光扫到", "sentence": "目光扫到桌角。", "example_fix": "交替切入"},
        ]
        msg = build_feedback_message(violations)
        assert "1." in msg
        assert "2." in msg


class TestStyleGuardLoop:
    @pytest.mark.asyncio
    async def test_clean_response_returns_unchanged(self):
        """无违规的响应应原样返回。"""
        result = {
            "choices": [{"message": {"role": "assistant", "content": "他点了点头。"}}]
        }

        async def fake_upstream():
            raise AssertionError("不应调用上游重发")

        final = await apply_style_guard_loop(
            body={"messages": []},
            call_upstream=fake_upstream,
            result=result,
            rules=RULES,
            max_retries=2,
        )
        assert final is result

    @pytest.mark.asyncio
    async def test_violation_triggers_retry(self):
        """含违规的响应应触发重发。"""
        first = {
            "choices": [{"message": {"role": "assistant", "content": "他坐在那里，没有动。"}}]
        }
        second = {
            "choices": [{"message": {"role": "assistant", "content": "他坐在那里，双手搁在桌面。"}}]
        }
        call_count = 0

        async def fake_upstream():
            nonlocal call_count
            call_count += 1
            return second

        body = {"messages": [
            {"role": "user", "content": "写一段叙事"}
        ]}

        final = await apply_style_guard_loop(
            body=body,
            call_upstream=fake_upstream,
            result=first,
            rules=RULES,
            max_retries=2,
        )
        assert call_count == 1
        assert final["choices"][0]["message"]["content"] == "他坐在那里，双手搁在桌面。"

    @pytest.mark.asyncio
    async def test_tool_calls_are_stripped_and_retry_continues(self):
        """原始响应含 tool_calls + 文本违规：剥离 tool_calls 后修正循环继续。"""
        first = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "他坐在那里，没有动。",
                "tool_calls": [
                    {"id": "tc1", "type": "function",
                     "function": {"name": "Edit", "arguments": "{}"}}
                ],
            }}]
        }
        second = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "他坐在那里，双手搁在桌面。",
                "tool_calls": [
                    {"id": "tc1", "type": "function",
                     "function": {"name": "Edit", "arguments": "{}"}}
                ],
            }}]
        }
        call_count = 0

        async def fake_upstream():
            nonlocal call_count
            call_count += 1
            return second

        body = {"messages": [
            {"role": "user", "content": "写一段叙事"}
        ]}

        final = await apply_style_guard_loop(
            body=body,
            call_upstream=fake_upstream,
            result=first,
            rules=RULES,
            max_retries=2,
        )
        assert call_count == 1, "tool_calls 剥离后应继续修正循环"
        final_msg = final["choices"][0]["message"]
        assert final_msg["content"] == "他坐在那里，双手搁在桌面。"
        assert final_msg["tool_calls"][0]["id"] == "tc1"

    @pytest.mark.asyncio
    async def test_tool_calls_in_corrected_result_merged_with_original(self):
        """修正后响应含 tool_calls：回滚到前一版并合并原始 tool_calls。"""
        first = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "他坐在那里，没有动。",
                "tool_calls": [
                    {"id": "tc1", "type": "function",
                     "function": {"name": "Edit", "arguments": "{}"}}
                ],
            }}]
        }
        # 重发后上游返回修正文本 + 全新 tool_calls（应丢弃，只保留原始 tool_calls）
        second = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "修正后的文本。",
                "tool_calls": [
                    {"id": "tc_new", "type": "function",
                     "function": {"name": "Bash", "arguments": "{}"}}
                ],
            }}]
        }
        call_count = 0

        async def fake_upstream():
            nonlocal call_count
            call_count += 1
            return second

        body = {"messages": [
            {"role": "user", "content": "写一段叙事"}
        ]}

        final = await apply_style_guard_loop(
            body=body,
            call_upstream=fake_upstream,
            result=first,
            rules=RULES,
            max_retries=2,
        )
        assert call_count == 1
        final_msg = final["choices"][0]["message"]
        # 修正后含 tool_calls → 回滚到 first，first 的 tool_calls 已在循环末尾合并
        assert final_msg["tool_calls"][0]["id"] == "tc1"

    @pytest.mark.asyncio
    async def test_no_violation_with_tool_calls_passes_through(self):
        """无违规 + 有 tool_calls：直接返回原 result，不调用上游。"""
        result = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "这段文本干净。",
                "tool_calls": [
                    {"id": "tc1", "type": "function",
                     "function": {"name": "Edit", "arguments": "{}"}}
                ],
            }}]
        }

        async def fake_upstream():
            raise AssertionError("不应调用上游重发")

        final = await apply_style_guard_loop(
            body={"messages": []},
            call_upstream=fake_upstream,
            result=result,
            rules=RULES,
            max_retries=2,
        )
        assert final is result

    @pytest.mark.asyncio
    async def test_tool_call_edit_args_with_violations_not_merged(self):
        """Edit tool 参数含违规时，原 tool_call 不合并回修正结果。"""
        import json as _json
        first = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "修正如下。",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {
                    "name": "Edit",
                    "arguments": _json.dumps({
                        "file_path": "test.md",
                        "new_string": "他站在那里，没有动。"
                    })
                }}],
            }}]
        }
        second = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "他站在那里，双手放在身体两侧。",
            }}]
        }
        call_count = 0

        async def fake_upstream():
            nonlocal call_count
            call_count += 1
            return second

        body = {"messages": [{"role": "user", "content": "写一段叙事"}]}

        final = await apply_style_guard_loop(
            body=body,
            call_upstream=fake_upstream,
            result=first,
            rules=RULES,
            max_retries=2,
        )
        assert call_count == 1
        # 参数违规→不合并原 tool_calls，保留 LLM 修正后的无 tool_calls 响应
        assert "tool_calls" not in final["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_tool_call_write_args_clean_merged_normally(self):
        """Write tool 参数无违规时，原 tool_call 正常合并。"""
        import json as _json
        result = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "文件写入完成。",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {
                    "name": "Write",
                    "arguments": _json.dumps({
                        "file_path": "test.md",
                        "content": "print('hello world')"
                    })
                }}],
            }}]
        }

        async def fake_upstream():
            raise AssertionError("不应调用上游重发")

        final = await apply_style_guard_loop(
            body={"messages": []},
            call_upstream=fake_upstream,
            result=result,
            rules=RULES,
            max_retries=2,
        )
        # 无违规→tool_calls 保留
        assert final["choices"][0]["message"]["tool_calls"][0]["id"] == "tc1"

    @pytest.mark.asyncio
    async def test_failed_retry_does_not_blank_original(self):
        """重发返回空/故障响应时，不得用空响应取代有内容的原始响应。"""
        original = {
            "choices": [{"message": {"role": "assistant", "content": "他坐在那里，没有动。"}}]
        }  # 1 个违规

        async def fake_upstream():
            return {"choices": []}  # 模拟上游瞬时故障

        final = await apply_style_guard_loop(
            body={"messages": [{"role": "user", "content": "写"}]},
            call_upstream=fake_upstream,
            result=original,
            rules=RULES,
            max_retries=2,
        )
        # 返回原始（有内容、虽有瑕疵），而非被清空
        assert final is original
        assert final["choices"][0]["message"]["content"] == "他坐在那里，没有动。"

    @pytest.mark.asyncio
    async def test_body_messages_restored_after_loop(self):
        """循环结束后 body['messages'] 应回滚到入口状态（与流式路径一致）。"""
        first = {"choices": [{"message": {"role": "assistant", "content": "他坐着，没有动。"}}]}
        clean = {"choices": [{"message": {"role": "assistant", "content": "他坐着，翻开卷宗。"}}]}

        async def fake_upstream():
            return clean

        body = {"messages": [{"role": "user", "content": "写"}]}
        await apply_style_guard_loop(
            body=body, call_upstream=fake_upstream, result=first, rules=RULES, max_retries=2,
        )
        assert len(body["messages"]) == 1  # 入口只有 1 条 user，修正轮已回滚

    @pytest.mark.asyncio
    async def test_content_violation_preserves_clean_tool_calls(self):
        """prose 违规 + 干净 tool_calls：即使重发改写成纯文本（丢 tool_calls），也须回挂原始。"""
        first = {"choices": [{"message": {
            "role": "assistant", "content": "他坐着，没有动。",
            "tool_calls": [{"id": "w1", "type": "function",
                            "function": {"name": "Write", "arguments": "{}"}}],
        }}]}
        # 重发只改了 prose，没有带回 tool_calls
        plain = {"choices": [{"message": {"role": "assistant",
                                          "content": "他坐着，翻开了卷宗。"}}]}
        calls = 0

        async def fake_upstream():
            nonlocal calls
            calls += 1
            return plain

        final = await apply_style_guard_loop(
            body={"messages": [{"role": "user", "content": "写"}]},
            call_upstream=fake_upstream, result=first, rules=RULES, max_retries=2,
        )
        assert calls == 1
        msg = final["choices"][0]["message"]
        assert msg["content"] == "他坐着，翻开了卷宗。"
        assert msg["tool_calls"][0]["id"] == "w1"  # 干净工具调用被保留

    @pytest.mark.asyncio
    async def test_override_on_retry_returns_stripped_result(self):
        """override 标签出现在重试响应上时，返回（已剥离标签的）该响应，而非更早候选。"""
        first = {"choices": [{"message": {"role": "assistant", "content": "他坐着，没有动。"}}]}
        override_resp = {
            "choices": [{"message": {"role": "assistant",
                                     "content": "[style-override]他不看也不应。"}}]
        }

        async def fake_upstream():
            return override_resp

        final = await apply_style_guard_loop(
            body={"messages": [{"role": "user", "content": "写"}]},
            call_upstream=fake_upstream,
            result=first,
            rules=RULES,
            max_retries=2,
        )
        assert final is override_resp
        assert "[style-override]" not in final["choices"][0]["message"]["content"]
        assert final["choices"][0]["message"]["content"] == "他不看也不应。"

    @pytest.mark.asyncio
    async def test_returns_best_when_retries_worse(self):
        """重试结果比原始更差时，返回原始（最优）而非最后一次。"""
        original = {
            "choices": [{"message": {"role": "assistant", "content": "他坐在那里，没有动。"}}]
        }  # 1 个违规 (fr5)
        worse = {
            "choices": [{"message": {"role": "assistant",
                                     "content": "他没有动，没有说话，目光扫到桌角。"}}]
        }  # 多个违规

        async def fake_upstream():
            return worse

        final = await apply_style_guard_loop(
            body={"messages": [{"role": "user", "content": "写"}]},
            call_upstream=fake_upstream,
            result=original,
            rules=RULES,
            max_retries=2,
        )
        # 原始违规数 < 重试违规数 → 返回原始
        assert final is original


class TestScanToolCallViolations:
    """tool_call 参数（Edit.new_string / Write.content）违规扫描。"""

    def _tc(self, name: str, args: dict):
        import json
        return [{"id": "tc1", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]

    def test_edit_new_string_violation(self):
        hits = scan_tool_call_violations(self._tc("Edit", {"new_string": "他站在那里，没有动。"}))
        assert any(h["rule_id"] == "fr5" for h in hits)

    def test_write_content_violation(self):
        hits = scan_tool_call_violations(self._tc("Write", {"content": "他走向办公室——里面坐了人。"}))
        assert any(h["rule_id"] == "dash" for h in hits)

    def test_clean_tool_args_no_violation(self):
        hits = scan_tool_call_violations(self._tc("Write", {"content": "print('hello')"}))
        assert hits == []

    def test_non_target_tool_ignored(self):
        hits = scan_tool_call_violations(self._tc("Bash", {"command": "他没有动"}))
        assert hits == []

    def test_non_dict_json_args_does_not_crash(self):
        # arguments 是合法 JSON 但非对象（null / 数字）—— 旧实现会 AttributeError
        assert scan_tool_call_violations(
            [{"function": {"name": "Write", "arguments": "null"}}]) == []
        assert scan_tool_call_violations(
            [{"function": {"name": "Edit", "arguments": "123"}}]) == []

    def test_malformed_json_args_skipped(self):
        assert scan_tool_call_violations(
            [{"function": {"name": "Write", "arguments": "{not json"}}]) == []

    def test_none_and_empty_tool_calls(self):
        assert scan_tool_call_violations(None) == []
        assert scan_tool_call_violations([]) == []


class TestNarrativeAnchors:
    """_has_narrative_anchors 必须在真实中文叙事文本上命中（修复 \\b CJK bug）。"""

    def test_narrative_text_detected(self):
        text = "他走到门前，转身看向窗外。她坐下，茶杯放在桌上。夜色里，码头的灯光很暗。"
        assert _has_narrative_anchors(text) is True

    def test_code_text_not_detected(self):
        assert _has_narrative_anchors("def foo():\n    return sum(range(10))") is False

    def test_common_chinese_not_false_positive(self):
        # 裸单字锚点已移除：常见非叙事词（时光/声明/部门/光阴）不应误判为叙事
        text = "时光荏苒，声明部门发布了声明，光阴似箭，门口的声响。"
        assert _has_narrative_anchors(text) is False

    def test_non_string_safe(self):
        assert _has_narrative_anchors(None) is False
        assert _has_narrative_anchors("") is False


class TestFluencyToolCalls:
    """apply_fluency_fix 对含 tool_calls 的响应须审查工具写入的叙事正文并就地回写，
    同时**保留 tool_calls**（这是 agent 写小说到文件的核心场景，不可整体跳过）。"""

    @pytest.mark.asyncio
    async def test_fluency_reviews_tool_written_content_and_preserves_call(self):
        import json
        from deep_proxy.optimization.style_guard import apply_fluency_fix
        # Write.content 是叙事正文（含充足锚点）；message.content 为简短状态（无锚点）
        narrative = "他走到门前，转身看向窗外。她坐下，茶杯放在桌上。夜色里，码头很暗。"
        result = {"choices": [{"message": {
            "role": "assistant", "content": "已写入文件。",
            "tool_calls": [{"id": "t1", "type": "function", "function": {
                "name": "Write",
                "arguments": json.dumps({"file_path": "ch1.md", "content": narrative}),
            }}],
        }}]}

        async def fake_upstream():
            # fluency 审查返回改写后的正文
            return {"choices": [{"message": {"content": narrative + "（已润色）"}}]}

        out = await apply_fluency_fix(
            {"messages": [{"role": "user", "content": "写"}]}, fake_upstream, result,
        )
        msg = out["choices"][0]["message"]
        # 工具调用保留
        assert msg["tool_calls"][0]["id"] == "t1"
        # 工具写入正文被就地润色
        new_args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert new_args["content"] == narrative + "（已润色）"
        assert new_args["file_path"] == "ch1.md"  # 其余参数不变

    @pytest.mark.asyncio
    async def test_fluency_skips_non_narrative_tool_args(self):
        import json
        from deep_proxy.optimization.style_guard import apply_fluency_fix
        # Write.content 是代码（无叙事锚点）→ 不审查、不改写
        result = {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "t1", "type": "function", "function": {
                "name": "Write",
                "arguments": json.dumps({"file_path": "a.py", "content": "def f():\n    return 1"}),
            }}],
        }}]}
        called = 0

        async def fake_upstream():
            nonlocal called
            called += 1
            return {"choices": [{"message": {"content": "x"}}]}

        out = await apply_fluency_fix({"messages": []}, fake_upstream, result)
        assert called == 0  # 无叙事锚点 → 不触发审查
        assert json.loads(out["choices"][0]["message"]["tool_calls"][0]
                          ["function"]["arguments"])["content"] == "def f():\n    return 1"

    @pytest.mark.asyncio
    async def test_fluency_missing_messages_key_safe(self):
        """body 缺 messages（形态异常）时直接返回原 result，不 KeyError。"""
        from deep_proxy.optimization.style_guard import apply_fluency_fix
        result = {"choices": [{"message": {"role": "assistant",
            "content": "他走到门前，转身看向窗外。她坐下，茶杯放在桌上。夜色里，码头很暗。"}}]}

        async def fake_upstream():
            raise AssertionError("不应调用上游")

        out = await apply_fluency_fix({}, fake_upstream, result)
        assert out is result


class TestAllRulesHaveIds:
    def test_all_rules_present(self):
        # 29 original + 5 first repo batch + 4 second repo batch + 2 third repo batch = 40
        assert len(RULES) == 47
        seen = set()
        for r in RULES:
            assert isinstance(r, StyleRule)
            assert r.id not in seen, f"Duplicate rule id: {r.id}"
            assert r.pattern_name, f"Empty pattern_name for {r.id}"
            assert r.example_fix, f"Empty example_fix for {r.id}"
            seen.add(r.id)


class TestRepoRules:
    """来自 DeepSeek V4 用户反馈的高频句式检测。"""

    def test_rp1_formulaic_closure(self):
        assert scan_violations("她看着窗外，这就够了。")  # has "这就够了"
        assert scan_violations("那就够了。")

    def test_rp2_ai_emotional_cliche(self):
        assert scan_violations("我稳稳接住你的不安。")

    def test_rp3_summary_marker(self):
        assert scan_violations("总而言之，一切都会好起来。")
        assert scan_violations("综上所述，我们需要继续前行。")

    def test_rp4_essay_template(self):
        assert scan_violations("这是成功的基石。")
        assert scan_violations("那是成长中的必修课。")

    def test_rp5_frozen_micro_expression(self):
        assert scan_violations("他眨了眨眼，没说话。")
        assert scan_violations("喉结上下滚动了一下。")

    def test_attr_a_also_matches_feibing(self):
        """attr_a 现在也匹配「并非…而是…」变体。"""
        assert scan_violations("他并非不在意，而是选择了沉默。")

    def test_rp6_template_simile(self):
        assert scan_violations("语气平淡，像在说今天的天气一样。")
        assert scan_violations("语气平淡得仿佛在说别人的事。")

    def test_rp7_triple_negation_parallelism(self):
        assert scan_violations("不笑，不说话，不看他一眼。")
        assert scan_violations("不躲，不藏，不绕。")

    def test_rp8_contradiction(self):
        assert scan_violations("很近，但很远。")
        assert scan_violations("很爱很爱，但很痛。")

    def test_rp9_romantic_closure(self):
        assert scan_violations("或许这就是命运的安排。")
        assert scan_violations("大概这就是成长的意义吧。")

    def test_rp10_double_negation_template(self):
        assert scan_violations("夜风无影无踪地穿过走廊。")
        assert scan_violations("她无悲无喜地看着他。")

    def test_rp11_god_perspective_causal(self):
        assert scan_violations("他之所以选择沉默，是因为不想再争辩了。")


class TestQuoteViolation:
    """quote_violation=False 的高风险模式不在反馈中引用原文。"""

    def test_no_quote_for_high_risk_patterns(self):
        violations = [
            {
                "rule_id": "attr_a",
                "pattern_name": "归因句式",
                "match_text": "不是温暖的笑，而是幸福的笑",
                "sentence": "她笑了。不是温暖的笑。而是幸福的笑。",
                "example_fix": "直接写行为",
                "quote_violation": False,
            },
        ]
        msg = build_feedback_message(violations)
        # 不应引用违规原文（防止负面表达注入强化）
        assert "不是温暖的笑" not in msg
        assert "归因句式是禁用的写作模式" in msg
        assert "直接写行为" in msg

    def test_quote_for_low_risk_patterns(self):
        """quote_violation=True（默认）的正常模式仍引用原文。"""
        violations = [
            {
                "rule_id": "a5",
                "pattern_name": "半寸/几息",
                "sentence": "推了半寸。",
                "example_fix": "动词重叠式",
                "quote_violation": True,
            },
        ]
        msg = build_feedback_message(violations)
        assert "推了半寸" in msg
