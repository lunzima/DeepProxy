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
    pick_one,
    pick_n,
)
from deep_proxy.utils import prepend_to_system_message as prepend_to_system
from deep_proxy.router import DeepProxyRouter


class TestSillyPoolContent:
    """池内容质量举证——不做穷举遍历。"""

    def test_spotcheck_quality(self):
        assert len(SILLY_PRIMING_POOL) == 8
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
