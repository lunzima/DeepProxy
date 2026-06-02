"""DynamicThresholdController 单元测试。

控制器：闭环反馈，把"阈值驱动升格率"驱动到 target=1-flash_floor，
因子 f 以 1.0 为中心、在 [1-band, 1+band] 内浮动。
"""
from __future__ import annotations

from deep_proxy.optimization.dynamic_threshold import DynamicThresholdController


def test_warmup_returns_unity_factor_below_min_samples():
    """样本数 < min_samples 时 f 恒为 1.0（暖机，不调整）。"""
    c = DynamicThresholdController(min_samples=10, window=50)
    for _ in range(9):
        c.record(True)  # 即便全升格，暖机期也不动 f
    assert c.factor == 1.0


def test_factor_rises_when_upgrade_rate_too_high():
    """升格率远超 target（flash 跌破 floor）→ f > 1.0（抬高阈值压制升格）。"""
    c = DynamicThresholdController(
        flash_floor=0.40, band=0.20, window=20, kp=0.5, min_samples=10,
    )
    for _ in range(20):
        c.record(True)  # upgrade_rate = 1.0, target = 0.60, error = +0.40
    # f = clamp(1 + 0.5*0.40, 0.8, 1.2) = clamp(1.2, ...) = 1.2
    assert c.factor == 1.2


def test_factor_drops_when_upgrade_rate_too_low():
    """升格率低于 target（flash 高于 floor）→ f < 1.0（降低阈值多升格）。"""
    c = DynamicThresholdController(
        flash_floor=0.40, band=0.20, window=20, kp=0.5, min_samples=10,
    )
    for _ in range(20):
        c.record(False)  # upgrade_rate = 0.0, target = 0.60, error = -0.60
    # f = clamp(1 + 0.5*(-0.60), 0.8, 1.2) = clamp(0.7, ...) = 0.8
    assert c.factor == 0.8


def test_factor_near_unity_at_equilibrium():
    """升格率恰在 target（60%）→ f ≈ 1.0（尊重配置阈值）。"""
    c = DynamicThresholdController(
        flash_floor=0.40, band=0.20, window=10, kp=0.5, min_samples=10,
    )
    for _ in range(6):
        c.record(True)
    for _ in range(4):
        c.record(False)  # upgrade_rate = 6/10 = 0.60 = target
    assert abs(c.factor - 1.0) < 1e-9


def test_window_evicts_old_decisions():
    """滑动窗口只看最近 window 个决策（旧决策被驱逐）。"""
    c = DynamicThresholdController(
        flash_floor=0.40, band=0.20, window=10, kp=0.5, min_samples=10,
    )
    for _ in range(10):
        c.record(True)   # 填满窗口全 True
    assert c.factor == 1.2
    for _ in range(10):
        c.record(False)  # 再 10 个 False 把 True 全挤出
    assert c.factor == 0.8


def test_min_samples_exceeding_window_rejected():
    """min_samples > window 会让控制器永久暖机（inert）→ config 层应拒绝。"""
    import pytest
    from pydantic import ValidationError
    from deep_proxy.config import DynamicThresholdConfig
    with pytest.raises(ValidationError):
        DynamicThresholdConfig(window=10, min_samples=20)


def test_samples_property_counts_window():
    """samples 反映当前窗口样本数（供 health / 测试观测）。"""
    c = DynamicThresholdController(window=5, min_samples=10)
    assert c.samples == 0
    for _ in range(3):
        c.record(True)
    assert c.samples == 3
    for _ in range(10):
        c.record(False)
    assert c.samples == 5  # 被 maxlen 截断


def test_factor_clamped_within_band():
    """无论误差多大，f 始终钳制在 [1-band, 1+band]。"""
    c = DynamicThresholdController(
        flash_floor=0.40, band=0.10, window=20, kp=5.0, min_samples=10,
    )
    for _ in range(20):
        c.record(True)   # 巨大正误差 × 大 kp
    assert c.factor == 1.10
    for _ in range(20):
        c.record(False)
    assert c.factor == 0.90
