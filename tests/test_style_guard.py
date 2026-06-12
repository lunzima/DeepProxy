"""测试 StyleGuard — 扫描 + 反馈 + 重发循环。"""

from __future__ import annotations

import pytest

from deep_proxy.optimization.style_guard import (
    RULES,
    StyleRule,
    _extract_sentence,
    apply_style_guard_loop,
    build_feedback_message,
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


class TestAllRulesHaveIds:
    def test_all_rules_present(self):
        # 29 original + 5 first repo batch + 4 second repo batch + 2 third repo batch = 40
        assert len(RULES) == 40
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
