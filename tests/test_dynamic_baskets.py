"""测试动态短段注入：硬规则 + 拼接器 + router 集成。"""

from __future__ import annotations

import random
from typing import List

from deep_proxy.config import (
    CreativeSamplingConfig,
    OptimizationConfig,
    PreciseSamplingConfig,
    ProxyConfig,
)
from deep_proxy.optimization.dynamic_baskets import (
    _BASKET_ORDER as BASKET_ORDER,
    _CODING_BASKETS as CODING_BASKETS,
    _CREATIVE_BASKETS as CREATIVE_BASKETS,
    _GENERAL_BASKETS as GENERAL_BASKETS,
    append_to_system,
    assemble_paragraph,
    scenario_from_profile,
)
from deep_proxy.router import DeepProxyRouter


class TestAssemblePara:
    def test_three_sentences_in_fixed_order(self):
        rng = random.Random(0)
        para = assemble_paragraph("coding", rng=rng)
        parts: List[str] = [p for p in para.split("。") if p]
        assert len(parts) == 3
        assert parts[0] in {s.rstrip("。") for s in CODING_BASKETS["methodology"]}
        assert parts[1] in {s.rstrip("。") for s in CODING_BASKETS["best_practices"]}
        assert parts[2] in {s.rstrip("。") for s in CODING_BASKETS["moderate_encouragement"]}

    def test_unknown_scenario_returns_empty(self):
        assert assemble_paragraph("nonexistent") == ""

    def test_writing_creative_picks_from_creative_baskets(self):
        rng = random.Random(0)
        para = assemble_paragraph("writing", writing_kind="creative", rng=rng)
        parts: List[str] = [p for p in para.split("。") if p]
        assert len(parts) == 3
        assert parts[0] in {s.rstrip("。") for s in CREATIVE_BASKETS["methodology"]}
        assert parts[1] in {s.rstrip("。") for s in CREATIVE_BASKETS["best_practices"]}
        assert parts[2] in {s.rstrip("。") for s in CREATIVE_BASKETS["moderate_encouragement"]}

    def test_writing_general_picks_from_general_baskets(self):
        rng = random.Random(0)
        para = assemble_paragraph("writing", writing_kind="general", rng=rng)
        parts: List[str] = [p for p in para.split("。") if p]
        assert len(parts) == 3
        assert parts[0] in {s.rstrip("。") for s in GENERAL_BASKETS["methodology"]}
        assert parts[1] in {s.rstrip("。") for s in GENERAL_BASKETS["best_practices"]}
        assert parts[2] in {s.rstrip("。") for s in GENERAL_BASKETS["moderate_encouragement"]}

    def test_unknown_writing_kind_falls_back_to_creative(self):
        rng = random.Random(0)
        para = assemble_paragraph("writing", writing_kind="unknown", rng=rng)
        parts: List[str] = [p for p in para.split("。") if p]
        assert parts[0] in {s.rstrip("。") for s in CREATIVE_BASKETS["methodology"]}

    def test_empty_basket_returns_empty(self):
        # 容错路径：篮为空 → 整段返回 ""（避免半段注入）
        from deep_proxy.optimization import dynamic_baskets as db
        original = db._CREATIVE_BASKETS["methodology"]
        try:
            db._CREATIVE_BASKETS["methodology"] = []
            assert assemble_paragraph("writing", writing_kind="creative") == ""
        finally:
            db._CREATIVE_BASKETS["methodology"] = original


class TestScenarioFromProfile:
    def test_precise_maps_to_coding(self):
        assert scenario_from_profile(PreciseSamplingConfig()) == "coding"

    def test_creative_maps_to_writing(self):
        assert scenario_from_profile(CreativeSamplingConfig()) == "writing"

    def test_none_returns_none(self):
        assert scenario_from_profile(None) is None


class TestAppendToSystem:
    def test_appends_to_existing_system(self):
        msgs = [
            {"role": "system", "content": "ORIGINAL"},
            {"role": "user", "content": "hi"},
        ]
        append_to_system(msgs, "ADDED")
        assert msgs[0]["content"].endswith("ADDED")
        assert "ORIGINAL" in msgs[0]["content"]

    def test_inserts_when_no_system(self):
        msgs = [{"role": "user", "content": "hi"}]
        append_to_system(msgs, "ADDED")
        assert msgs[0] == {"role": "system", "content": "ADDED"}

    def test_inserts_new_when_system_is_multimodal(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "ORIGINAL"}]},
            {"role": "user", "content": "hi"},
        ]
        append_to_system(msgs, "ADDED")
        assert msgs[0] == {"role": "system", "content": "ADDED"}
        assert isinstance(msgs[1]["content"], list)


def _minimal_config(*, dynamic_baskets: bool = True) -> ProxyConfig:
    cfg = ProxyConfig()
    cfg.deepseek.api_key = "sk-test"
    cfg.optimization = OptimizationConfig(
        enabled=True,
        compress_skills=False,
        avoid_negative_style=False,
        assume_good_intent=False,
        instruction_priority=False,
        independent_analysis=False,
        reason_genuinely=False,
        inject_date=False,
        cot_reset=False,
        show_math_steps=False,
        prefer_multiple_sources=False,
        avoid_fabricated_citations=False,
        json_mode_hint=False,
        safe_inlined_content=False,
        re2=False,
        cot_reflection=False,
        readurls=False,
        dynamic_baskets=dynamic_baskets,
        silly_expert_priming=False,
    )
    return cfg


class TestRouterIntegration:
    async def test_coding_profile_appends_paragraph(self):
        router = DeepProxyRouter(_minimal_config())
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = await router.prepare_request(
            body, sampling_profile=PreciseSamplingConfig(),
        )
        sys_msgs = [m for m in out["messages"] if m.get("role") == "system"]
        assert sys_msgs, "应注入一条 system 消息"
        # 注入内容必须出自 coding 篮
        all_coding = (
            CODING_BASKETS["methodology"]
            + CODING_BASKETS["best_practices"]
            + CODING_BASKETS["moderate_encouragement"]
        )
        text = sys_msgs[0]["content"]
        assert any(s.rstrip("。") in text for s in all_coding)
        await router.close()

    async def test_writing_profile_creative_kind(self):
        cfg = _minimal_config()
        cfg.optimization.writing_basket_kind = "creative"
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = await router.prepare_request(
            body, sampling_profile=CreativeSamplingConfig(),
        )
        sys_msgs = [m for m in out["messages"] if m.get("role") == "system"]
        assert sys_msgs, "应注入一条 system 消息"
        all_creative = sum(CREATIVE_BASKETS.values(), [])
        text = sys_msgs[0]["content"]
        assert any(s.rstrip("。") in text for s in all_creative)
        await router.close()

    async def test_writing_profile_general_kind(self):
        cfg = _minimal_config()
        cfg.optimization.writing_basket_kind = "general"
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = await router.prepare_request(
            body, sampling_profile=CreativeSamplingConfig(),
        )
        sys_msgs = [m for m in out["messages"] if m.get("role") == "system"]
        assert sys_msgs
        all_general = sum(GENERAL_BASKETS.values(), [])
        text = sys_msgs[0]["content"]
        assert any(s.rstrip("。") in text for s in all_general)
        await router.close()

    async def test_disabled_skips(self):
        router = DeepProxyRouter(_minimal_config(dynamic_baskets=False))
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = await router.prepare_request(
            body, sampling_profile=PreciseSamplingConfig(),
        )
        assert not any(m.get("role") == "system" for m in out["messages"])
        await router.close()

    async def test_skipped_when_tools_present(self):
        router = DeepProxyRouter(_minimal_config())
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        }
        out = await router.prepare_request(
            body, sampling_profile=PreciseSamplingConfig(),
        )
        assert not any(m.get("role") == "system" for m in out["messages"])
        await router.close()

    async def test_appends_after_user_system(self):
        router = DeepProxyRouter(_minimal_config())
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "USER_SYSTEM"},
                {"role": "user", "content": "hi"},
            ],
        }
        out = await router.prepare_request(
            body, sampling_profile=PreciseSamplingConfig(),
        )
        sys_text = out["messages"][0]["content"]
        assert sys_text.startswith("USER_SYSTEM")
        assert len(sys_text) > len("USER_SYSTEM")
        await router.close()
