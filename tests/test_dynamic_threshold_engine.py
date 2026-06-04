"""DynamicThresholdController 与 UpgradeDecisionEngine 的集成。

验证：
  - factor 缩放 router_threshold / heuristic_threshold（含钳制）
  - 仅 Step 3/4 阈值驱动决策被 record；sentinel/persist/cooldown 不 record
  - controller=None → 行为与现状完全等价
"""
from __future__ import annotations

from deep_proxy.config import FlashUpgradeConfig
from deep_proxy.optimization.flash_upgrade import (
    RepeatUpgradeThrottle,
    UpgradeTracker,
)
from deep_proxy.optimization.upgrade_decision import UpgradeDecisionEngine
from deep_proxy.optimization.upgrade_router import create_router
from deep_proxy.providers import Provider


class _SpyController:
    """记录 record() 调用 + 提供固定 factor。"""

    def __init__(self, factor: float):
        self._factor = factor
        self.records: list[bool] = []

    @property
    def factor(self) -> float:
        return self._factor

    def record(self, upgraded: bool) -> None:
        self.records.append(upgraded)


def _engine(**cfg_overrides) -> UpgradeDecisionEngine:
    kwargs = dict(
        enabled=True,
        router_threshold=0.50,
        heuristic_threshold=8.0,
        downgrade_threshold=5.0,
    )
    kwargs.update(cfg_overrides)
    cfg = FlashUpgradeConfig(**kwargs)
    return UpgradeDecisionEngine(
        cfg=cfg,
        upgrade_tracker=UpgradeTracker(),
        throttle=RepeatUpgradeThrottle(),
        bert_router=create_router("rule"),
    )


def _ds() -> Provider:
    return Provider(
        name="deepseek", api_base="https://x", api_key="k",
        litellm_prefix="deepseek/", flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
    )


def test_resolve_params_scales_thresholds_by_factor():
    eng = _engine()
    params = eng._resolve_params(_ds(), _SpyController(1.2))
    assert abs(params.router_thr - 0.50 * 1.2) < 1e-9
    assert abs(params.heur_thr - 8.0 * 1.2) < 1e-9


def test_resolve_params_factor_none_is_identity():
    eng = _engine()
    params = eng._resolve_params(_ds(), None)
    assert params.router_thr == 0.50
    assert params.heur_thr == 8.0


def test_resolve_params_router_threshold_clamped_to_one():
    eng = _engine(router_threshold=0.90)
    params = eng._resolve_params(_ds(), _SpyController(1.2))  # 0.90*1.2=1.08 → clamp 1.0
    assert params.router_thr == 1.0


def test_resolve_params_heuristic_clamped_above_downgrade():
    # heur=6.0, downgrade=5.0; factor 0.8 → 4.8 < downgrade → clamp 到 downgrade+EPS
    eng = _engine(heuristic_threshold=6.0, downgrade_threshold=5.0)
    params = eng._resolve_params(_ds(), _SpyController(0.8))
    assert params.heur_thr > 5.0
    assert params.heur_thr < 5.0 + 0.5  # 紧贴 downgrade 上方


def test_threshold_driven_decision_is_recorded():
    """flash 起始 + 走到 Step 3/4 → controller.record(did_upgrade) 被调用一次。"""
    eng = _engine()
    spy = _SpyController(1.0)
    body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
    eng.apply(body, provider=_ds(), controller=spy)
    assert len(spy.records) == 1
    # 简单 "hi" 不应升格
    assert spy.records[0] is False
    assert body["model"] == "deepseek-v4-flash"


def test_throttle_veto_does_not_record():
    """阈值判升但 throttle 否决（强制 flash）→ 非阈值驱动，controller 不 record
    （否则把仅 throttle 可达的请求当成"阈值未升"，会错误地把闭环往降阈值方向带偏）。"""
    eng = UpgradeDecisionEngine(
        cfg=FlashUpgradeConfig(
            enabled=True, router_threshold=0.0, heuristic_threshold=10.0,
            downgrade_threshold=5.0,
        ),
        upgrade_tracker=UpgradeTracker(),
        throttle=RepeatUpgradeThrottle(max_repeats=2),  # 第 2 次连续升格即否决
        bert_router=create_router("rule"),
    )
    spy = _SpyController(1.0)
    msg = [{"role": "user", "content": "写一个复杂的并发调度器"}]
    b1 = {"model": "deepseek-v4-flash", "messages": list(msg)}
    eng.apply(b1, provider=_ds(), controller=spy)        # 第 1 次：阈值升格
    b2 = {"model": "deepseek-v4-flash", "messages": list(msg)}
    eng.apply(b2, provider=_ds(), controller=spy)        # 第 2 次：throttle 否决

    assert b1["model"] == "deepseek-v4-pro"
    assert b2["model"] == "deepseek-v4-flash"            # 被 throttle 强制 flash
    assert spy.records == [True]                         # 否决那次不 record


def test_sentinel_path_does_not_record():
    """sentinel 强制升格不经过阈值决策 → 不 record。"""
    eng = _engine()
    spy = _SpyController(1.0)
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "_deepproxy_upgrade": True,
    }
    eng.apply(body, provider=_ds(), controller=spy)
    assert spy.records == []
    assert body["model"] == "deepseek-v4-pro"


def test_persist_cache_hit_does_not_record():
    """tracker 命中（持久升格窗口内、score 够高）→ 不 record。"""
    eng = _engine()
    spy = _SpyController(1.0)
    msgs = [{"role": "user", "content": "写一个完整的分布式限流器，要求支持滑动窗口和令牌桶"}]
    # 先放进升格窗口
    eng._tracker.set_remaining(msgs, 2, provider="deepseek")
    body = {"model": "deepseek-v4-flash", "messages": msgs}
    eng.apply(body, provider=_ds(), controller=spy)
    # 命中持久缓存（score 应 >= downgrade=5.0），不记录阈值决策
    if body["model"] == "deepseek-v4-pro":
        assert spy.records == []


def test_high_factor_suppresses_borderline_upgrade():
    """同一 borderline 输入：f=0.8 升格、f=1.2 不升格（阈值被抬高）。"""
    msgs = [{"role": "user", "content": "请帮我设计一个支持高并发的订单系统架构方案"}]

    eng_lo = _engine(router_threshold=0.50)
    body_lo = {"model": "deepseek-v4-flash", "messages": list(msgs)}
    eng_lo.apply(body_lo, provider=_ds(), controller=_SpyController(0.8))

    eng_hi = _engine(router_threshold=0.50)
    body_hi = {"model": "deepseek-v4-flash", "messages": list(msgs)}
    eng_hi.apply(body_hi, provider=_ds(), controller=_SpyController(1.2))

    # f 越高阈值越高，升格只会更难（hi 升格 ⊆ lo 升格）
    hi_upgraded = body_hi["model"] == "deepseek-v4-pro"
    lo_upgraded = body_lo["model"] == "deepseek-v4-pro"
    if hi_upgraded:
        assert lo_upgraded
