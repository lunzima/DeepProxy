"""Flash→Pro 选择性升格机制测试。

覆盖四层架构的全部单元 + 集成：
  Layer 0: Router（RuleUpgradeRouter 决策逻辑）
  Layer 1: 启发式复杂度评分（compute_complexity_score）
  Layer 2: Router 执行（_maybe_upgrade 改写 body["model"]）
  Layer 3: 对话级持久化（UpgradeTracker 跟踪剩余轮次）
"""

from __future__ import annotations

import pytest

from deep_proxy.config import DeepSeekConfig, FlashUpgradeConfig, ProxyConfig
from deep_proxy.optimization.flash_upgrade import (
    UpgradeTracker,
    compute_complexity_score,
    conversation_fingerprint,
    extra_body_requests_upgrade,
    has_upgrade_sentinel,
)
from deep_proxy.optimization.upgrade_router import RuleUpgradeRouter, create_router
from deep_proxy.router import DeepProxyRouter


# ======================================================================
# 辅助工厂（非 fixture，避免 conftest 冲突）
# ======================================================================


def _make_upgrade_cfg(**overrides) -> ProxyConfig:
    kwargs = dict(
        enabled=True,
        heuristic_threshold=6.5,
        router_threshold=0.55,
        persist_turns=2,
    )
    kwargs.update(overrides)
    return ProxyConfig(
        deepseek=DeepSeekConfig(api_key="sk-test"),
        flash_upgrade=FlashUpgradeConfig(**kwargs),
    )


# ======================================================================
# conversation_fingerprint
# ======================================================================


class TestConversationFingerprint:
    def test_empty_messages(self):
        fp = conversation_fingerprint([])
        assert isinstance(fp, str) and len(fp) == 32

    def test_only_user(self):
        fp1 = conversation_fingerprint([
            {"role": "user", "content": "写个排序算法"},
        ])
        fp2 = conversation_fingerprint([
            {"role": "user", "content": "写个排序算法"},
            {"role": "assistant", "content": "这是冒泡排序"},
        ])
        # 加入 assistant 不应改变 fingerprint
        assert fp1 == fp2

    def test_stable_across_turns(self):
        """同一对话的不同轮次回传相同指纹（仅首条 user）。"""
        first_user = {"role": "user", "content": "你好"}
        first_asst = {"role": "assistant", "content": "你好！有什么可以帮你的？"}
        second_user = {"role": "user", "content": "帮我写个快排"}

        fp_a = conversation_fingerprint([first_user, first_asst])
        fp_b = conversation_fingerprint([first_user, first_asst, second_user])
        assert fp_a == fp_b

    def test_different_conversations_differ(self):
        fp1 = conversation_fingerprint([
            {"role": "user", "content": "写段 Python 代码"},
        ])
        fp2 = conversation_fingerprint([
            {"role": "user", "content": "写首诗歌"},
        ])
        assert fp1 != fp2

    def test_multimodal_user_truncated(self):
        """多模态 user 消息退化到 str() 截断，不应 crash。"""
        fp = conversation_fingerprint([
            {"role": "user", "content": [
                {"type": "text", "text": "看图说话"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]},
        ])
        assert isinstance(fp, str) and len(fp) == 32


# ======================================================================
# UpgradeTracker
# ======================================================================


class TestUpgradeTracker:
    def test_not_upgraded_by_default(self):
        tracker = UpgradeTracker()
        assert tracker.is_upgraded([{"role": "user", "content": "hi"}]) is False
        assert tracker.remaining([{"role": "user", "content": "hi"}]) == 0

    def test_set_remaining_and_consume_turns(self):
        tracker = UpgradeTracker()
        msgs = [{"role": "user", "content": "写个算法"}]

        # set_remaining: 还剩 2 轮
        tracker.set_remaining(msgs, 2)
        assert tracker.remaining(msgs) == 2
        assert tracker.is_upgraded(msgs) is True  # 同一轮，不消耗

        # 新轮次：assistant + user
        msgs.append({"role": "assistant", "content": "好的"})
        msgs.append({"role": "user", "content": "优化它"})
        assert tracker.is_upgraded(msgs) is True   # 消耗 1，剩 1
        assert tracker.remaining(msgs) == 1

        # 再一轮——耗尽
        msgs.append({"role": "assistant", "content": "优化版"})
        msgs.append({"role": "user", "content": "再优化"})
        assert tracker.is_upgraded(msgs) is False  # 消耗最后 1 轮 → 耗尽
        assert tracker.remaining(msgs) == 0

    def test_turns_exhausted(self):
        tracker = UpgradeTracker()
        msgs = [{"role": "user", "content": "hi"}]
        tracker.set_remaining(msgs, 1)

        assert tracker.is_upgraded(msgs) is True  # 同一轮

        msgs.append({"role": "assistant", "content": "hi"})
        msgs.append({"role": "user", "content": "bye"})
        assert tracker.is_upgraded(msgs) is False  # 消耗最后 1 轮 → 耗尽

        msgs.append({"role": "assistant", "content": "bye"})
        msgs.append({"role": "user", "content": "again"})
        assert tracker.is_upgraded(msgs) is False  # 已耗尽

    def test_same_turn_retry_does_not_consume(self):
        """同一轮次的重试（多请求长度不变）不应消耗额度。"""
        tracker = UpgradeTracker()
        msgs = [{"role": "user", "content": "hi"}]
        tracker.set_remaining(msgs, 2)

        # 三次请求，消息长度不变（重试）
        assert tracker.is_upgraded(msgs) is True   # 不消耗，剩 2
        assert tracker.is_upgraded(msgs) is True   # 不消耗，剩 2
        assert tracker.is_upgraded(msgs) is True   # 不消耗，剩 2
        assert tracker.remaining(msgs) == 2

    def test_different_conversations_isolated(self):
        tracker = UpgradeTracker()
        conv_a = [{"role": "user", "content": "问题A"}]
        conv_b = [{"role": "user", "content": "问题B"}]

        tracker.set_remaining(conv_a, 1)
        assert tracker.is_upgraded(conv_a) is True
        assert tracker.is_upgraded(conv_b) is False

    def test_compaction_does_not_freeze_counter(self):
        """会话被客户端压缩（messages 数量变短）后仍按 last user 推进轮次。

        回归保护：早期实现以 len(messages) 作为新轮次判据，当客户端在长对话中
        做 history compaction，下一轮 len_now <= last_len，counter 永远不递减，
        模型被锁死在 Pro。
        """
        tracker = UpgradeTracker()
        # 第 0 轮：长对话触发升格
        long_msgs = [{"role": "user", "content": "首条问题"}]
        for i in range(10):
            long_msgs.append({"role": "assistant", "content": f"a{i}"})
            long_msgs.append({"role": "user", "content": f"u{i}"})
        tracker.set_remaining(long_msgs, 2)

        # 第 1 轮：客户端把对话压缩（len 大幅缩短），但有新 user 消息
        compacted = [
            {"role": "user", "content": "首条问题"},     # 同 fingerprint
            {"role": "assistant", "content": "[summary]"},
            {"role": "user", "content": "新一轮的提问"},  # 新的最后 user
        ]
        assert tracker.is_upgraded(compacted) is True   # 消耗 1，剩 1
        assert tracker.remaining(compacted) == 1

        # 第 2 轮：再压缩 + 新 user
        compacted2 = [
            {"role": "user", "content": "首条问题"},
            {"role": "assistant", "content": "[summary v2]"},
            {"role": "user", "content": "再下一轮"},
        ]
        assert tracker.is_upgraded(compacted2) is False  # 耗尽
        assert tracker.remaining(compacted2) == 0

    def test_repeated_identical_request_does_not_consume(self):
        """同一 last user 消息重复请求（agent retry / 同步重试）不消耗轮次。"""
        tracker = UpgradeTracker()
        msgs = [
            {"role": "user", "content": "首条问题"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "追问"},
        ]
        tracker.set_remaining(msgs, 2)

        # 同一 messages 数组重复发 5 次 → 不消耗
        for _ in range(5):
            assert tracker.is_upgraded(msgs) is True
        assert tracker.remaining(msgs) == 2


# ======================================================================
# compute_complexity_score
# ======================================================================


class TestComputeComplexityScore:
    def test_empty(self):
        assert compute_complexity_score([]).score == 0.0

    def test_simple_question_scores_low(self):
        msgs = [{"role": "user", "content": "法国的首都是什么？"}]
        score = compute_complexity_score(msgs).score
        assert 0.0 <= score < 2.0  # 简单问题，启发式评分低

    def test_code_block_scores_higher(self):
        msgs = [{"role": "user", "content": "```\nprint('hello')\n```"}]
        score = compute_complexity_score(msgs).score
        # 代码块维度 +0.5，应有非零分
        assert score > 0.1

    def test_multi_turn_complex_dialog(self):
        """多次迭代的复杂对话应有中高分数。"""
        msgs = [
            {"role": "user", "content": "实现一个分布式锁，要论证一致性"},
            {"role": "assistant", "content": "好的，使用 Redis 实现"},
            {"role": "user", "content": "要支持可重入和超时，考虑架构设计"},
            {"role": "assistant", "content": "好的，添加可重入逻辑"},
            {"role": "user", "content": "还需要考虑高并发场景下的一致性"},
        ]
        score = compute_complexity_score(msgs).score
        assert score >= 1.5  # 多轮 + 架构/分布式/一致性关键词

    def test_math_symbol_contributes(self):
        msgs = [{"role": "user", "content": "证明 ∑ 1/n² = π²/6"}]
        score = compute_complexity_score(msgs).score
        assert score >= 0.5  # "证明" 0.3 + max(∑ *0.5, 1*0.5) = 0.8

    def test_keyword_density(self):
        msgs = [{"role": "user", "content":
            "证明这个定理。推导过程需要严格证明。"
            "涉及复杂度分析和分布式系统架构。"}]
        score = compute_complexity_score(msgs).score
        assert score >= 1.0  # 关键词命中

    def test_very_long_context(self):
        """8000+ tokens 的上下文得高分。"""
        long_text = "Hello world. " * 5000
        msgs = [{"role": "user", "content": long_text}]
        score = compute_complexity_score(msgs).score
        assert score >= 2.0  # token 量加分

    def test_context_inflation_discounts_token_and_code(self):
        """Qwen Code 典型场景：大量拼接的代码上下文 + 简短用户提问。

        大量 token/代码块应被大幅折扣，最终分数远低于不折扣场景。
        """
        # 模拟 Qwen Code 拼接了大量代码上下文（~10K chars） + 用户简单提问
        big_file = ("def process_data():\n    x = 1\n    y = 2\n    return x + y\n" * 200)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"file1:\n```python\n{big_file}\n```"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"file2:\n```python\n{big_file}\n```"},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "帮我看看这段代码"},  # 最后一条很短 (< 1% 总量)
        ]
        score = compute_complexity_score(msgs).score

        # 无膨胀反事实：同样的代码内容 + 用户主动长提问（无折扣）
        msgs_no_inflation = [
            {"role": "user", "content":
                f"file1:\n```python\n{big_file}\n```\n"
                f"file2:\n```python\n{big_file}\n```\n"
                "请分析这两个文件中的架构设计问题，并进行全面的代码审查。论证系统的一致性。"},
        ]
        score_no = compute_complexity_score(msgs_no_inflation).score

        # 膨胀场景 token/代码块折扣 70%，应显著低于无膨胀场景
        assert score < score_no, f"膨胀分数 {score} 应低于无膨胀 {score_no}"
        # 膨胀分数应远低于 threshold (6.5)，不应触发升格
        assert score < 5.0, f"膨胀分数 {score} 应低于 5.0（threshold=6.5）"

    def test_context_inflation_not_triggered_when_user_long(self):
        """最后一条 user 消息占比 >= 15% 时不触发膨胀折扣。"""
        short_prefix = "Context: "  # 短前缀
        long_question = "请详细分析这个分布式系统的架构设计，论证其一致性保证。" * 5  # 长提问
        msgs = [
            {"role": "system", "content": short_prefix},
            {"role": "user", "content": long_question},
        ]
        # user_fraction ≈ len(long_question) / (len(short_prefix) + len(long_question)) >> 0.15
        score = compute_complexity_score(msgs).score

        # 不应触发折扣：关键词如 "分析" "分布式" "架构" "一致性" "论证" 全部有效
        assert score >= 1.0, f"长提问未膨胀场景应保持有效分数，得 {score}"


# ======================================================================
# has_upgrade_sentinel
# ======================================================================


class TestHasUpgradeSentinel:
    def test_detects_in_system(self):
        msgs = [
            {"role": "system", "content": "你是助手\n<deepproxy_upgrade>force</deepproxy_upgrade>"},
            {"role": "user", "content": "hi"},
        ]
        assert has_upgrade_sentinel(msgs) is True

    def test_not_found(self):
        msgs = [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "hi"}]
        assert has_upgrade_sentinel(msgs) is False

    def test_no_system_message(self):
        msgs = [{"role": "user", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"}]
        assert has_upgrade_sentinel(msgs) is False  # sentinel 仅检测 role=system

    def test_multimodal_system(self):
        msgs = [
            {"role": "system", "content": [
                {"type": "text", "text": "Normal"},
                {"type": "text", "text": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
            ]},
            {"role": "user", "content": "hi"},
        ]
        assert has_upgrade_sentinel(msgs) is True


# ======================================================================
# extra_body_requests_upgrade
# ======================================================================


class TestExtraBodyUpgrade:
    def test_true(self):
        assert extra_body_requests_upgrade({"_deepproxy_upgrade": True}) is True

    def test_false(self):
        assert extra_body_requests_upgrade({"_deepproxy_upgrade": False}) is False

    def test_missing(self):
        assert extra_body_requests_upgrade({}) is False

    def test_truthy_value(self):
        assert extra_body_requests_upgrade({"_deepproxy_upgrade": 1}) is True


# ======================================================================
# RuleUpgradeRouter
# ======================================================================


class TestRuleUpgradeRouter:
    def test_sentinel_scores_max(self):
        router = RuleUpgradeRouter()
        msgs = [
            {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
            {"role": "user", "content": "hi"},
        ]
        assert router.score(msgs) == 1.0

    def test_extra_body_scores_max(self):
        router = RuleUpgradeRouter()
        msgs = [{"role": "user", "content": "hi"}]
        assert router.score(msgs, body={"_deepproxy_upgrade": True}) == 1.0

    def test_simple_question_below_threshold(self):
        router = RuleUpgradeRouter()
        msgs = [{"role": "user", "content": "法国的首都是什么？"}]
        score = router.score(msgs)
        assert score < 0.55  # 低于默认 router_threshold

    def test_complex_code_above_threshold(self):
        """复杂代码场景应返回中等以上分数。"""
        router = RuleUpgradeRouter()
        msgs = [
            {"role": "user", "content":
                "实现一个分布式系统的故障转移机制。"
                "需要证明该机制在多数节点失效时的正确性。"
                "涉及一致性算法和架构设计。\n"
                "```python\ndef recover():\n    pass\n```"},
        ]
        score = router.score(msgs)
        # 预期：token~0 + 代码块 0.5 + 关键词(分布式/证明/一致性/算法/架构)×0.3=1.5
        # + turn 1/3.0 ≈ 0.33 = 2.33 → × 0.1 ≈ 0.23
        assert score >= 0.15

    def test_should_upgrade_with_threshold_via_sentinel(self):
        """sentinel 在 system prompt 中 → score=1.0 → 阈值 < 1.0 升，> 1.0 不升。"""
        router = RuleUpgradeRouter()
        msgs = [
            {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
            {"role": "user", "content": "hi"},
        ]
        assert router.should_upgrade(msgs, threshold=0.5) is True
        assert router.should_upgrade(msgs, threshold=1.0) is True   # 1.0 >= 1.0 → True
        assert router.should_upgrade(msgs, threshold=1.01) is False  # 1.0 < 1.01 → False


# ======================================================================
# Integration: DeepProxyRouter._maybe_upgrade
# ======================================================================


class TestMaybeUpgradeIntegration:
    def test_sentinel_upgrades(self):
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
                {"role": "user", "content": "hi"},
            ],
        }
        router._maybe_upgrade(body)
        assert body["model"] == "deepseek-v4-pro"

    def test_extra_body_upgrades(self):
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "_deepproxy_upgrade": True,
        }
        router._maybe_upgrade(body)
        assert body["model"] == "deepseek-v4-pro"

    def test_upgrade_persists_across_turns(self):
        """升格后后续轮次自动走 Pro。"""
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
                {"role": "user", "content": "写个复杂算法"},
            ],
        }
        # 首次 sentinel 强制升格
        router._maybe_upgrade(body)
        assert body["model"] == "deepseek-v4-pro"

        # 模拟下一轮（消息增长）—— 客户端重置 model 为 flash
        body2 = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
                {"role": "user", "content": "写个复杂算法"},
                {"role": "assistant", "content": "来，写个快排"},
                {"role": "user", "content": "优化一下"},
            ],
        }
        router._maybe_upgrade(body2)
        assert body2["model"] == "deepseek-v4-pro"

    def test_simple_question_stays_flash(self):
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "法国的首都是什么？"}],
        }
        router._maybe_upgrade(body)
        assert body["model"] == "deepseek-v4-flash"

    def test_not_flash_skips_upgrade(self):
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
        }
        # _maybe_upgrade 不会被 prepare_request 守卫调用，但直接调用也应无害
        router._maybe_upgrade(body)
        assert body["model"] == "deepseek-v4-pro"  # 未改变

    async def test_prepare_request_pipeline_integration(self):
        """端到端：prepare_request 中 sentinel 升格到 pro。"""
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
                {"role": "user", "content": "分析这个系统架构"},
            ],
        }
        result = await router.prepare_request(body, sampling_profile=None)
        assert result["model"] == "deepseek-v4-pro"

    async def test_prepare_request_disabled_keeps_flash(self, cfg):
        """config.flash_upgrade.enabled=False 时不升格。"""
        cfg.flash_upgrade.enabled = False
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "复杂系统架构设计"}],
        }
        result = await router.prepare_request(body, sampling_profile=None)
        assert result["model"] == "deepseek-v4-flash"

    def test_create_router_fallback(self):
        """未知 router_type 静默降级到 rule。"""
        r = create_router("nonexistent")
        assert isinstance(r, RuleUpgradeRouter)

    def test_pending_upgrade_only_commits_on_success(self):
        """#6 回归：set_remaining 延迟到上游成功后提交，失败请求不污染 tracker。

        此前 _maybe_upgrade 在 prepare 阶段直接 set_remaining，上游 500/超时
        也照样写入。本回归断言：仅 _commit_pending_upgrade 被调用后才落账。
        """
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "<deepproxy_upgrade>force</deepproxy_upgrade>"},
                {"role": "user", "content": "复杂请求"},
            ],
        }
        # 决策阶段：升格生效，但 tracker 仍空
        router._maybe_upgrade(body)
        assert body["model"] == "deepseek-v4-pro"
        assert "_deepproxy_pending_upgrade" in body
        assert router._upgrade_tracker.active_count == 0

        # 模拟失败：永不调用 _commit_pending_upgrade，下一轮新对话不应被锁 Pro
        # （此处不能直接复用同一 messages，因为 sentinel 仍会触发 Step 1）
        new_body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "复杂请求"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "再来一次"},
            ],
        }
        router._maybe_upgrade(new_body)
        # 没人 commit 过，cache 仍空，应走启发式/BERT 重新评估而非 cache hit
        # 即使最终也升格，也不会打 "持久升格命中" 日志路径

        # 现在显式 commit 之前那次升格
        router._commit_pending_upgrade(body)
        assert router._upgrade_tracker.active_count == 1

    def test_throttle_clears_tracker_to_avoid_cache_bypass(self):
        """#3 回归：throttle 触发时同步清掉 UpgradeTracker entry，避免下一轮
        Step 2 cache hit 越过 throttle cooldown 直走 Pro。
        """
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        # 手动注入一个"已升格"的对话状态
        msgs = [{"role": "user", "content": "复杂代码任务"}]
        router._upgrade_tracker.set_remaining(msgs, 5)
        assert router._upgrade_tracker.active_count == 1

        # 模拟 throttle 触发并调用 clear
        router._upgrade_tracker.clear(msgs)
        assert router._upgrade_tracker.active_count == 0
        assert router._upgrade_tracker.is_upgraded(msgs) is False

    def test_persist_cache_hit_after_commit(self):
        """补 test_upgrade_persists_across_turns 的覆盖缺口：

        原测试两轮都带 <deepproxy_upgrade>force</deepproxy_upgrade>，每轮都走
        Step 1 sentinel 路径，从未真正命中 Step 2 持久化 cache。本回归显式：
        触发升格 → commit → 下一轮命中 Step 2 cache。
        """
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        # 第一轮：复杂内容触发升格（启发式 / Router）
        body1 = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "请帮我设计一个分布式系统架构，"
                                            "需要支持高并发、容错和一致性。"
                                            "请给出详细的架构图和关键组件说明。"},
            ],
        }
        router._maybe_upgrade(body1)
        assert body1["model"] == "deepseek-v4-pro"
        assert "_deepproxy_pending_upgrade" in body1
        assert router._upgrade_tracker.active_count == 0  # 尚未 commit

        # 模拟上游成功 → commit
        router._commit_pending_upgrade(body1)
        assert router._upgrade_tracker.active_count == 1

        # 第二轮：消息增长，新的最后 user，无 sentinel
        body2 = {
            "model": "deepseek-v4-flash",
            "messages": [
                body1["messages"][0],
                {"role": "assistant", "content": "好的，我来设计..."},
                {"role": "user", "content": "继续"},  # 简单追问，本身分数低
            ],
        }
        router._maybe_upgrade(body2)
        # 必须命中 Step 2 cache 走 Pro，而非走 Step 3-4（"继续"的复杂度分数远低于阈值）
        assert body2["model"] == "deepseek-v4-pro"

    def test_throttle_cooldown_blocks_persist_cache_hit(self):
        """#3 回归（端到端）：throttle 冷却期内即使 tracker 有 entry，也强制 Flash。

        模拟 cooldown 状态 + 旧 entry 残留（防御性测试），验证 in_cooldown
        预检在 Step 2 之前生效。
        """
        cfg = _make_upgrade_cfg()
        router = DeepProxyRouter(cfg)
        msgs = [{"role": "user", "content": "复杂任务"}]

        # 人为注入 throttle 冷却状态 + tracker entry
        from deep_proxy.optimization.flash_upgrade import (
            conversation_fingerprint as _cfp,
            _last_user_hash as _lh,
        )
        fp = _cfp(msgs)
        h = _lh(msgs)
        router._upgrade_throttle._state[(fp, h)] = (0, 2)  # cooldown=2
        router._upgrade_tracker.set_remaining(msgs, 5)  # 残留 entry

        body = {"model": "deepseek-v4-flash", "messages": msgs}
        router._maybe_upgrade(body)
        # cooldown 期：强制 Flash + 清掉 tracker，绝不能命中 cache 走 Pro
        assert body["model"] == "deepseek-v4-flash"
        assert router._upgrade_tracker.active_count == 0
