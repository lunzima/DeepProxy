"""测试无厘头 expert priming：质量举证 + prepend + router 集成。"""

from __future__ import annotations

from deep_proxy.config import (
    CreativeSamplingConfig,
    OptimizationConfig,
    PreciseSamplingConfig,
    ProxyConfig,
)
from deep_proxy.optimization.silly_priming import (
    SILLY_PRIMING_POOL,
    _AMD_ATTRIBUTION,
    _AMD_SENTENCE,
    _NEUTRAL_ATTRIBUTIONS,
    pick_one,
    pick_n,
    wrap_for_injection,
)
from deep_proxy.utils import prepend_to_system_message as prepend_to_system
from deep_proxy.router import DeepProxyRouter


class TestSillyPoolContent:
    """池内容质量举证——不做穷举遍历。"""

    def test_spotcheck_quality(self):
        assert len(SILLY_PRIMING_POOL) == 20
        for s in SILLY_PRIMING_POOL:
            assert s.endswith("。"), f"非陈述句末: {s}"
        # 代表性举例——否定字不应出现在专家预置池中
        assert not any("不" in s for s in SILLY_PRIMING_POOL), "池中含否定字"


class TestPickN:
    def test_pick_two_returns_two_unique_items(self):
        items = pick_n(2)
        assert len(items) == 2
        assert items[0] != items[1]
        for s in items:
            assert s in SILLY_PRIMING_POOL

    def test_pick_n_respects_rng(self):
        import random
        rng = random.Random(0)
        items_a = pick_n(2, rng=rng)
        rng = random.Random(0)
        items_b = pick_n(2, rng=rng)
        assert items_a == items_b

    def test_pick_n_exceeding_pool_size_returns_all(self):
        items = pick_n(100)
        assert len(items) == len(SILLY_PRIMING_POOL)
        assert set(items) == set(SILLY_PRIMING_POOL)

    def test_pick_one_still_works(self):
        item = pick_one()
        assert item in SILLY_PRIMING_POOL


class TestWrapForInjection:
    """注入包装：每条独立成段+独立署名，AMD 句固定署名，其余从中性库抽。"""

    def test_empty_returns_empty_string(self):
        assert wrap_for_injection([]) == ""

    def test_amd_sentence_uses_amd_attribution(self):
        out = wrap_for_injection([_AMD_SENTENCE])
        assert _AMD_SENTENCE in out
        assert _AMD_ATTRIBUTION in out

    def test_non_amd_uses_neutral_attribution(self):
        non_amd = next(s for s in SILLY_PRIMING_POOL if s != _AMD_SENTENCE)
        out = wrap_for_injection([non_amd])
        assert non_amd in out
        # 必须命中中性库中的一个署名
        assert any(attr in out for attr in _NEUTRAL_ATTRIBUTIONS)
        # AMD 署名不能出现在非 AMD 条目上
        assert _AMD_ATTRIBUTION not in out

    def test_each_item_gets_attribution_line(self):
        non_amd = [s for s in SILLY_PRIMING_POOL if s != _AMD_SENTENCE][:2]
        out = wrap_for_injection(non_amd)
        # 段间空行隔开 → 包含 \n\n
        assert "\n\n" in out
        # 两条均出现
        assert all(s in out for s in non_amd)
        # 至少出现 2 个 "—— "
        assert out.count("—— ") >= 2

    def test_mixed_amd_and_neutral(self):
        non_amd = next(s for s in SILLY_PRIMING_POOL if s != _AMD_SENTENCE)
        out = wrap_for_injection([_AMD_SENTENCE, non_amd])
        assert _AMD_ATTRIBUTION in out
        assert any(attr in out for attr in _NEUTRAL_ATTRIBUTIONS)
        assert _AMD_SENTENCE in out
        assert non_amd in out

    def test_no_silly_label_leak(self):
        """整块文本不得出现暴露 silly 性质的关键词。"""
        items = pick_n(2)
        out = wrap_for_injection(items)
        forbidden = ("弱智", "金句", "段子", "搞笑", "测试", "实验", "扰动")
        for word in forbidden:
            assert word not in out, f"包装文本不得含 '{word}'"

    def test_rng_makes_attribution_deterministic(self):
        import random
        non_amd = [s for s in SILLY_PRIMING_POOL if s != _AMD_SENTENCE][:2]
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        assert wrap_for_injection(non_amd, rng=rng_a) == wrap_for_injection(non_amd, rng=rng_b)


class TestPrependToSystem:
    def test_prepends_to_existing_system(self):
        msgs = [
            {"role": "system", "content": "ORIGINAL"},
            {"role": "user", "content": "hi"},
        ]
        prepend_to_system(msgs, "PRIMER")
        assert msgs[0]["content"].startswith("PRIMER")
        assert msgs[0]["content"].endswith("ORIGINAL")

    def test_inserts_when_no_system(self):
        msgs = [{"role": "user", "content": "hi"}]
        prepend_to_system(msgs, "PRIMER")
        assert msgs[0] == {"role": "system", "content": "PRIMER"}

    def test_inserts_new_when_system_is_multimodal(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "ORIG"}]},
            {"role": "user", "content": "hi"},
        ]
        prepend_to_system(msgs, "PRIMER")
        assert msgs[0] == {"role": "system", "content": "PRIMER"}
        assert isinstance(msgs[1]["content"], list)


def _minimal_config(*, silly: bool = True) -> ProxyConfig:
    cfg = ProxyConfig()
    cfg.deepseek.api_key = "sk-test"
    cfg.optimization = OptimizationConfig(
        enabled=True,
        compress_skills=False,
        avoid_negative_style=False,
        natural_temperament=False,
        contextual_register=False,
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
        dynamic_baskets=False,
        silly_expert_priming=silly,
    )
    return cfg


class TestRouterIntegration:
    async def test_silly_priming_prepended_when_enabled(self):
        router = DeepProxyRouter(_minimal_config(silly=True))
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "USER_SYSTEM"},
                {"role": "user", "content": "hi"},
            ],
        }
        out = await router.prepare_request(body, sampling_profile=PreciseSamplingConfig())
        sys_text = out["messages"][0]["content"]
        # 注入 2 条 priming，首条在最前面，次条紧随其后，均来自池
        assert any(sys_text.startswith(p) for p in SILLY_PRIMING_POOL)
        assert "USER_SYSTEM" in sys_text
        pool = set(SILLY_PRIMING_POOL)
        injected = [p for p in pool if p in sys_text]
        assert len(injected) >= 2, f"应注入 2 条 priming，实际找到: {injected}"
        await router.close()

    async def test_silly_priming_disabled(self):
        router = DeepProxyRouter(_minimal_config(silly=False))
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = await router.prepare_request(body, sampling_profile=PreciseSamplingConfig())
        # 关闭后不注入任何 system 消息
        assert not any(m.get("role") == "system" for m in out["messages"])
        await router.close()

    async def test_silly_priming_skipped_with_tools(self):
        router = DeepProxyRouter(_minimal_config(silly=True))
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        }
        out = await router.prepare_request(body, sampling_profile=PreciseSamplingConfig())
        assert not any(m.get("role") == "system" for m in out["messages"])
        await router.close()

    async def test_silly_priming_works_under_creative_profile(self):
        """全场景生效：creative profile 也应注入 2 条。"""
        router = DeepProxyRouter(_minimal_config(silly=True))
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = await router.prepare_request(body, sampling_profile=CreativeSamplingConfig())
        sys_msgs = [m for m in out["messages"] if m.get("role") == "system"]
        assert sys_msgs
        sys_text = sys_msgs[0]["content"]
        pool = set(SILLY_PRIMING_POOL)
        injected = [p for p in pool if p in sys_text]
        assert len(injected) >= 2, f"creative profile 应注入 2 条 priming，实际找到: {injected}"
        await router.close()
