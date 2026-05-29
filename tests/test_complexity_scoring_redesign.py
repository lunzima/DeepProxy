"""Complexity scoring 重设测试（spec: 2026-05-29-complexity-scoring-redesign.md）。

5 维评分：keyword + math + turn + last_user_size + reasoning_density
+ Direction C hysteresis（router._maybe_upgrade Step 2）。

覆盖三方向：
- A: 简单 user + 复杂 assistant grind → reasoning_score 累积升格
- B: 复杂 user + 简单 follow-up → keyword + reasoning 累积保 Pro
- C: 触发 Pro 后机械重复 → reasoning_density 跌零 → hysteresis 主动降格
"""
from __future__ import annotations

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.optimization.flash_upgrade import compute_complexity_score


def _simple_cfg():
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek", "api_base": "x", "api_key": "y",
                "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash",
                "pro_model": "deepseek-v4-pro",
            },
        },
        "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
        "deepseek": {"api_key": "y"},
        "flash_upgrade": {
            "enabled": True,
            "router_type": "rule",
            "router_threshold": 0.65,
            "heuristic_threshold": 8.0,
            "downgrade_threshold": 3.0,
        },
    })


# ---------------------------------------------------------------------------
# 5 维各自贡献
# ---------------------------------------------------------------------------


def test_empty_messages_returns_zero():
    assert compute_complexity_score([]).score == 0.0


def test_keyword_dim_caps_at_2_0():
    # 大量关键词命中应被 cap 在 2.0
    text = "证明 算法 复杂度 数据结构 分布式 一致性 重构 架构 系统设计 优化 " * 5
    msgs = [{"role": "user", "content": text}]
    score = compute_complexity_score(msgs).score
    # keyword cap 2.0 + last_user_size ~0.6 + turn 0.33 ≈ 2.9+
    assert score >= 2.5


def test_math_dim_user_only():
    msgs = [{"role": "user", "content": "∑ ∫ ∂ ∇ ∈ ∉ ⊂ ⊃"}]  # 8 数学符号 × 0.5 = 4.0 → cap 1.5
    result = compute_complexity_score(msgs)
    # math = 1.5；其它信号很小
    assert result.score >= 1.5


def test_turn_dim():
    msgs = []
    for i in range(6):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    # 6 user turns → turn_score = min(6/3, 2.0) = 2.0
    score = compute_complexity_score(msgs).score
    assert score >= 2.0


def test_last_user_size_dim():
    long_q = "x" * 900  # 900 字 → cap 3.0
    msgs = [{"role": "user", "content": long_q}]
    score = compute_complexity_score(msgs).score
    assert score >= 3.0  # last_user_size alone


def test_reasoning_density_dim_zero_when_no_assistant():
    msgs = [{"role": "user", "content": "hi"}]
    # 无 assistant → reasoning_score = 0
    result = compute_complexity_score(msgs)
    # 仅 turn 0.33 + last_user_size 0.007 ≈ 0.34
    assert result.score < 0.5


def test_reasoning_density_dim_accumulates_with_v4_thinking():
    """V4 thinking 模式下 assistant 带 reasoning_content → 长度直接驱动信号。"""
    msgs = [{"role": "user", "content": "hi"}]
    for _ in range(3):
        msgs.append({
            "role": "assistant", "content": "ok",
            "reasoning_content": "x" * 2000,  # 每条 2000 字 reasoning
        })
    # avg = 2000 chars/turn → reasoning_score = min(2000/500, 4.0) = 4.0 (cap)
    score = compute_complexity_score(msgs).score
    assert score >= 4.0


# ---------------------------------------------------------------------------
# 方向 A: 简单 user prompt + 复杂 assistant grind 应升格
# ---------------------------------------------------------------------------


def test_direction_a_simple_user_plus_long_reasoning_grind_high_score():
    msgs = [{"role": "user", "content": "做"}]
    for _ in range(8):
        msgs.append({
            "role": "assistant", "content": "继续工作中...",
            "reasoning_content": "长篇推理：分析问题，分解步骤，验证假设。" * 200,  # ~5000+ chars
        })
    score = compute_complexity_score(msgs).score
    # reasoning avg ~5000 chars/turn → score = min(5000/500, 8.0) = 8.0 cap
    # 加 turn 0.33 → 总分应 >= 8
    assert score >= 8.0, f"长 reasoning grind 应抬高 score 到 heuristic 阈值，实际 {score}"


def test_direction_a_end_to_end_triggers_heuristic_upgrade():
    """Direction A 端到端：简单 user + 8 轮长 reasoning grind → router 实际升格。"""
    from deep_proxy.router import DeepProxyRouter
    cfg = _simple_cfg()  # heuristic_threshold = 8.0
    router = DeepProxyRouter(cfg)

    # 简单 user prompt，无关键词 — 模拟典型 Direction A 场景
    msgs = [{"role": "user", "content": "做"}]
    for _ in range(8):
        msgs.append({
            "role": "assistant",
            "content": "继续工作中...",
            "reasoning_content": "深度推理：分析问题，分解步骤，验证假设。" * 200,
        })

    body = {"model": "deepseek-v4-flash", "messages": msgs}
    router._maybe_upgrade(body, provider=cfg.providers["deepseek"])

    # reasoning_score 单维度 cap=8.0 已能跨过 heuristic_threshold（启发式路径）
    assert body["model"] == "deepseek-v4-pro", (
        f"Direction A 长 reasoning grind 应被 heuristic 升格，实际 {body['model']}"
    )


# ---------------------------------------------------------------------------
# 方向 B: 复杂初始 user + 简单 follow-up "继续" 仍保高分
# ---------------------------------------------------------------------------


def test_direction_b_complex_user_simple_followup_score_stable():
    """复杂 user keyword 已累积 + assistant 已有 reasoning + 简单 follow-up。

    last_user_size 会因为 follow-up 短而低（"继续" → 2 字），但 keyword（累积全部 user）
    和 reasoning 都不会丢，总分仍应 >= downgrade_threshold (3.0)。
    """
    msgs = [
        {"role": "user", "content": "证明算法 复杂度 分布式 一致性 架构 优化"},
        {"role": "assistant", "content": "好的", "reasoning_content": "深度推理..." * 100},
        {"role": "user", "content": "继续"},
    ]
    score = compute_complexity_score(msgs).score
    # keyword ~ 6 hits * 0.3 = 1.8（user 累积包括"继续"，但"继续"不命中关键词）
    # math 0, turn 2/3=0.67, last_user_size 2/300=0.007, reasoning avg 大 → 4.0 cap
    # 总分 ≈ 6.5 ≫ downgrade_thr=3.0
    assert score >= 3.0, f"复杂上下文应保高分不骤降，实际 {score}"


# ---------------------------------------------------------------------------
# 方向 C: 触发 Pro 后机械重复 → 主动降格（router 层 hysteresis）
# ---------------------------------------------------------------------------


def test_reasoning_score_is_windowed_not_lifetime_average():
    """reasoning_score 必须用最近 N 轮滑动窗口，不能用全历史平均。

    回归用户实测发现：长 agent loop 早期深度 reasoning 让全历史平均值永久居高
    （score 钉死在 8-10），即便后续转向机械重复也降不下来 → Direction C
    主动降格永远触发不了 → Pro 锁定到 hash 改变。

    场景：50 轮深度 reasoning + 3 轮机械化（reasoning=空）
      - 全历史平均：(50×4000 + 3×0) / 53 ≈ 3774 字 → score=7.5（仍极高）
      - 最近 3 轮窗口：(0+0+0) / 3 = 0 字 → score=0（Direction C 可触发）
    """
    msgs = [{"role": "user", "content": "task"}]
    # 50 轮深度推理
    for _ in range(50):
        msgs.append({
            "role": "assistant", "content": "...",
            "reasoning_content": "x" * 4000,
        })
    # 3 轮机械化（无 reasoning_content）
    for _ in range(3):
        msgs.append({"role": "assistant", "content": "ok"})

    score = compute_complexity_score(msgs).score
    # 关键断言：windowed → reasoning_score ≈ 0；总分应 << downgrade_threshold(3.0)
    # 全历史平均会让 reasoning_score ≈ 7.5（错误行为）
    # 其它维度：keyword(0) + math(0) + turn(0.33) + last_user_size(~0.01) + reasoning(0)
    # 总分 ≈ 0.34，well below 3.0
    assert score < 3.0, (
        f"最近 3 轮 reasoning=0 应让 score 跌破 downgrade_threshold，"
        f"实际 {score}（若为旧全历史平均，score 应 ~7.5）"
    )


def test_reasoning_score_window_recovers_on_recent_deep_turns():
    """对照：最近 3 轮重新深度推理 → reasoning_score 立即回升。

    确认窗口"前向"也敏感——不会因历史平庸而压住 score。
    """
    msgs = [{"role": "user", "content": "x"}]
    # 历史 20 轮极轻 reasoning
    for _ in range(20):
        msgs.append({
            "role": "assistant", "content": "ok",
            "reasoning_content": "短",  # 仅 1 字
        })
    # 最近 3 轮深度
    for _ in range(3):
        msgs.append({
            "role": "assistant", "content": "...",
            "reasoning_content": "x" * 4500,
        })
    score = compute_complexity_score(msgs).score
    # 最近 3 轮 avg=4500 → reasoning_score=min(4500/500, 8.0)=8.0 (cap)
    # 即便历史全是 1 字 reasoning，windowed 也只看最近 3 轮
    assert score >= 8.0, (
        f"最近 3 轮深度 reasoning 应让 score 达到 reasoning cap，"
        f"实际 {score}（若为旧全历史平均，会被 20 轮短 reasoning 拖低）"
    )


def test_direction_c_active_downgrade_when_reasoning_dries_up():
    """升格后 assistant 全是机械回复（无 reasoning_content）→ score 跌破阈值 → clear tracker。"""
    from deep_proxy.router import DeepProxyRouter
    cfg = _simple_cfg()
    router = DeepProxyRouter(cfg)

    # 构造场景：之前轮已触发升格（人为塞 tracker），本轮 assistant 全机械、无推理
    msgs = [{"role": "user", "content": "简单"}]
    for _ in range(3):
        msgs.append({"role": "assistant", "content": "ok"})  # 无 reasoning_content
    router._upgrade_tracker.set_remaining(msgs, 2, provider="deepseek")

    body = {"model": "deepseek-v4-flash", "messages": list(msgs)}
    router._maybe_upgrade(body, provider=cfg.providers["deepseek"])

    # score 极低：keyword=0, math=0, turn=0.33, last_user_size~0.007, reasoning=0
    # 总分 ~0.34 < downgrade_thr=3.0 → 主动撤销 + 走 flash
    assert body["model"] == "deepseek-v4-flash", (
        f"机械任务应被 Direction C 主动降格，实际 {body['model']}"
    )
    assert router._upgrade_tracker.remaining(msgs, provider="deepseek") == 0


def test_direction_c_keeps_pro_when_reasoning_still_high():
    """对照：升格后 assistant 仍持续高 reasoning_content → 维持 Pro。"""
    from deep_proxy.router import DeepProxyRouter
    cfg = _simple_cfg()
    router = DeepProxyRouter(cfg)

    msgs = [{"role": "user", "content": "x"}]
    for _ in range(3):
        msgs.append({
            "role": "assistant", "content": "...",
            "reasoning_content": "深度推理 " * 400,  # ~2000 chars/turn
        })
    router._upgrade_tracker.set_remaining(msgs, 2, provider="deepseek")

    body = {"model": "deepseek-v4-flash", "messages": list(msgs)}
    router._maybe_upgrade(body, provider=cfg.providers["deepseek"])

    # reasoning_score = min(2000/500, 4.0) = 4.0 (cap) ≫ downgrade_thr=3.0
    # 持久升格命中 → 维持 Pro
    assert body["model"] == "deepseek-v4-pro", (
        f"持续高 reasoning 应保 Pro，实际 {body['model']}"
    )


# ---------------------------------------------------------------------------
# per_provider downgrade_threshold 覆盖
# ---------------------------------------------------------------------------


def test_misconfig_downgrade_geq_heuristic_rejected():
    """downgrade_threshold >= heuristic_threshold 触发 pydantic 校验失败。"""
    with pytest.raises(ValueError, match="hysteresis"):
        ProxyConfig.model_validate({
            "providers": {
                "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                             "litellm_prefix": "deepseek/",
                             "flash_model": "deepseek-v4-flash",
                             "pro_model": "deepseek-v4-pro"},
            },
            "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
            "deepseek": {"api_key": "y"},
            "flash_upgrade": {
                "heuristic_threshold": 5.0,
                "downgrade_threshold": 5.0,  # 等于 → 拒绝
            },
        })


def test_misconfig_per_provider_downgrade_geq_heuristic_rejected():
    """per_provider 也要遵守 hysteresis 约束。"""
    with pytest.raises(ValueError, match="per_provider"):
        ProxyConfig.model_validate({
            "providers": {
                "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "openai/",
                         "flash_model": "mimo-v2.5", "pro_model": "mimo-v2.5-pro"},
            },
            "ports": [{"port": 8001, "provider": "mimo", "sampling": "creative"}],
            "deepseek": {"api_key": "y"},
            "flash_upgrade": {
                "heuristic_threshold": 8.0,
                "downgrade_threshold": 3.0,
                "per_provider": {"mimo": {"downgrade_threshold": 9.0}},  # > heuristic
            },
        })


def test_per_provider_downgrade_threshold_override():
    cfg = ProxyConfig.model_validate({
        "providers": {
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/",
                     "flash_model": "mimo-v2.5", "pro_model": "mimo-v2.5-pro"},
        },
        "ports": [{"port": 8001, "provider": "mimo", "sampling": "creative"}],
        "deepseek": {"api_key": "y"},
        "flash_upgrade": {
            "enabled": True,
            "downgrade_threshold": 3.0,
            "per_provider": {"mimo": {"downgrade_threshold": 1.5}},
        },
    })
    assert cfg.flash_upgrade.threshold_for_provider("mimo", "downgrade_threshold") == 1.5
    assert cfg.flash_upgrade.threshold_for_provider("deepseek", "downgrade_threshold") == 3.0
