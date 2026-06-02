"""Per-port 动态阈值控制器（闭环反馈）。

把 BERT/启发式升格阈值在配置值 ±band 带内浮动，使**阈值驱动**的升格率收敛到
`target = 1 - flash_floor`（默认 0.60，即 40% flash 均衡），从而保证快速模型
份额不低于 flash_floor。

设计要点：
  - 母体：仅"阈值能左右其结果"的升格决策（Step 3/4 的启发式+BERT 路径）。
    sentinel/throttle/persist 命中与 pool 直接选中的 pro 都**不** record（见
    upgrade_decision.py 的记录点）。
  - 双向控制律（比例控制）：f 以 1.0 为中心，升格过多 → f>1.0 抬高阈值压制；
    升格不足 → f<1.0 降低阈值多升格。均衡在 upgrade_rate=target 时 f≈1.0。
  - 暖机：样本数 < min_samples 时 f=1.0，避免小样本剧烈抖动。
  - 进程内状态，无持久化（重启清零可接受）。
"""
from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class DynamicThresholdController:
    """单 port 的闭环阈值调整器。

    用法：
      factor → 读当前 f（施加到 router_threshold / heuristic_threshold）
      record(upgraded) → 记录一次阈值驱动决策并更新 f
    """

    def __init__(
        self,
        *,
        flash_floor: float = 0.40,
        band: float = 0.20,
        window: int = 50,
        kp: float = 0.5,
        min_samples: int = 10,
    ):
        self._target_upgrade_rate = 1.0 - flash_floor
        self._band = band
        self._kp = kp
        self._min_samples = min_samples
        self._window: deque[bool] = deque(maxlen=window)
        self._factor = 1.0

    @property
    def factor(self) -> float:
        """当前调整因子 f，钳制在 [1-band, 1+band]；暖机期返回 1.0。"""
        return self._factor

    @property
    def samples(self) -> int:
        """当前滑动窗口样本数（供 health / 观测）。"""
        return len(self._window)

    def record(self, upgraded: bool) -> None:
        """记录一次阈值驱动的升格决策（True=升格到 pro），随后更新 f。"""
        self._window.append(bool(upgraded))
        self._recompute()

    def _recompute(self) -> None:
        n = len(self._window)
        if n < self._min_samples:
            self._factor = 1.0
            return
        upgrade_rate = sum(self._window) / n
        error = upgrade_rate - self._target_upgrade_rate
        raw = 1.0 + self._kp * error
        lo, hi = 1.0 - self._band, 1.0 + self._band
        clamped = min(hi, max(lo, raw))
        self._factor = clamped
        if clamped in (lo, hi) and raw != clamped:
            logger.info(
                "动态阈值饱和: f=%.3f (raw=%.3f) upgrade_rate=%.3f target=%.3f",
                clamped, raw, upgrade_rate, self._target_upgrade_rate,
            )
