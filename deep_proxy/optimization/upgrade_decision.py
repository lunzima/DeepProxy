"""Flash→Pro 升格决策引擎 — 把 5 步策略从 router 抽出。

router.DeepProxyRouter._maybe_upgrade 之前内联 250 行 god method 的一部分，
混合了：参数解析 / sentinel 短路 / 持久缓存命中 + Direction C hysteresis /
启发式 / BERT / 防刷屏 throttle / pending commit 储存。本模块把决策路径切成
6 个 step 方法 + 1 个 apply 入口；router 仅保留 1 行 shim。

设计要点：
  - 同 router 内联实现的语义与行为完全等价（46 个 test_flash_upgrade 用例
    + 15 个 test_complexity_scoring_redesign 用例不需修改）
  - 不引入"纯函数 + 副作用分离"——trackers 的读取本身就有副作用
    （consume_turn / should_throttle 都改状态），强行做读写分离需要重写
    tracker 类，超出本次抽取范围
  - 收益：每个 step 自含责任、可独立 mock tracker 测试，新增 per-provider
    策略 / 新增 step（如未来的 sampling-aware 判定）有清晰挂点
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..deepseek_models import V4_FLASH, V4_PRO
from ..providers import Provider
from .flash_upgrade import (
    UpgradeTracker,
    compute_complexity_score,
    extra_body_requests_upgrade,
    has_upgrade_sentinel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProviderParams:
    """从 Provider + FlashUpgradeConfig 派生的所有阈值与模型名快照。

    把"按 provider 派生参数"集中到一个不可变对象，每个 step 拿现成的，
    避免 5 个 step 各自反复调 cfg.threshold_for_provider。
    """
    pro_model: str
    flash_model: str
    provider_name: str
    router_thr: float
    heur_thr: float
    downgrade_thr: float


class UpgradeDecisionEngine:
    """Flash→Pro 升格决策。

    apply(body, provider) 是唯一外部入口，等价替换 router._maybe_upgrade。

    决策顺序（短路）：
      Step 1: Sentinel / extra_body 强制升格
      Step 2a: 防刷屏 throttle cooldown 仍生效 → 强制 flash
      Step 2b: 持久缓存命中 + Direction C hysteresis 重评估
      Step 3: 启发式快速路径（Layer 1）
      Step 4: BERT/Rule Router（Layer 0）
      Step 5: 防刷屏 throttle 提交 + 升格落地 + pending stash
    """

    def __init__(self, cfg, upgrade_tracker, throttle, bert_router):
        # cfg = FlashUpgradeConfig（含 threshold_for_provider 方法）
        self._cfg = cfg
        self._tracker = upgrade_tracker
        self._throttle = throttle
        self._bert = bert_router

    # ------------------------------------------------------------------ entry

    def apply(self, body: Dict[str, Any], *, provider: Optional[Provider] = None) -> None:
        """根据 5 步策略 mutate body['model']；升格时同步 stash pending commit。

        与原 router._maybe_upgrade 行为完全等价（API 不变）。
        """
        messages = body.get("messages", [])
        params = self._resolve_params(provider)

        # 短路链：任一 step 返回 True 表示已 mutate body + 决定完成
        if self._step_sentinel(body, messages, params):
            return
        if self._step_throttle_cooldown(messages, params):
            return
        # 持久缓存分两步：
        # (1) 检查是否在升格窗口内 + 重评估 → 若 score 低则 hysteresis 清缓存（不短路，让 step 3/4 重新决定）
        # (2) 若窗口内 + score 仍高 → 命中 cache，确定走 Pro 并短路
        # 拆两步避免单步既能"清+落+短路"又能"清+不短路"的混乱语义。
        if self._step_persist_cache_hit(body, messages, params):
            return
        self._step_compute_and_commit(body, messages, params)

    # ----------------------------------------------------------- step helpers

    def _resolve_params(self, provider: Optional[Provider]) -> _ProviderParams:
        cfg = self._cfg
        if provider is not None:
            return _ProviderParams(
                pro_model=provider.pro_model,
                flash_model=provider.flash_model,
                provider_name=provider.name,
                router_thr=cfg.threshold_for_provider(provider.name, "router_threshold"),
                heur_thr=cfg.threshold_for_provider(provider.name, "heuristic_threshold"),
                downgrade_thr=cfg.threshold_for_provider(provider.name, "downgrade_threshold"),
            )
        # provider=None 兜底：硬编码 V4 默认 + 全局阈值（与原实现一致）
        return _ProviderParams(
            pro_model=V4_PRO,
            flash_model=V4_FLASH,
            provider_name="deepseek",
            router_thr=cfg.router_threshold,
            heur_thr=cfg.heuristic_threshold,
            downgrade_thr=cfg.downgrade_threshold,
        )

    def _stash_pending(
        self, body: Dict[str, Any], messages: List[Dict[str, Any]],
        params: _ProviderParams,
    ) -> None:
        """在 body 写入延迟提交所需的 fingerprint + last_user_hash 快照。

        延迟提交避免失败请求白扣 Pro 槽位；快照在此处取因为 messages 后续
        会被 skills 阶段（RE2/readurls 等）改写。
        """
        fp, last_user_h = UpgradeTracker.snapshot_keys(messages)
        body["_deepproxy_pending_upgrade"] = {
            "fingerprint": fp,
            "last_user_hash": last_user_h,
            "turns": self._cfg.persist_turns,
            "provider": params.provider_name,
        }

    # ----------------------------------------------------------------- steps

    def _step_sentinel(
        self, body: Dict[str, Any], messages: List[Dict[str, Any]],
        params: _ProviderParams,
    ) -> bool:
        """Step 1：sentinel / extra_body 强制升格（最高优先级）。"""
        if has_upgrade_sentinel(messages) or extra_body_requests_upgrade(body):
            logger.info("Sentinel 强制升格 → %s", params.pro_model)
            body["model"] = params.pro_model
            self._stash_pending(body, messages, params)
            return True
        return False

    def _step_throttle_cooldown(
        self, messages: List[Dict[str, Any]], params: _ProviderParams,
    ) -> bool:
        """Step 2a：throttle 冷却期内必须强制 flash，阻断后续所有路径。

        throttle 触发后的 cooldown 期内不能让 persist cache 越过 throttle。
        同时主动清掉 tracker entry 避免下一轮 Step 2b 越过 throttle。
        """
        if not self._throttle.in_cooldown(messages):
            return False
        # 推进冷却计数 + 清残留持久升格
        self._throttle.should_throttle(messages, False)
        self._tracker.clear(messages, provider=params.provider_name)
        logger.info("升格限流冷却中 → 强制 %s", params.flash_model)
        return True  # 不 mutate body['model']（保持原 flash），但短路完成

    def _step_persist_cache_hit(
        self, body: Dict[str, Any], messages: List[Dict[str, Any]],
        params: _ProviderParams,
    ) -> bool:
        """Step 2b：持久缓存命中 + Direction C hysteresis 重评估。

        分支：
          - tracker 无升格记录 → 返回 False，让 step 3/4 正常决策
          - tracker 有升格记录 + 复杂度 < downgrade_thr → 清 tracker 后
            返回 False（主动 hysteresis 降格；继续走 step 3/4 重新决定）
          - tracker 有升格记录 + 复杂度 >= downgrade_thr → 写 pro_model
            到 body 并返回 True（短路）

        注：is_upgraded 在迁移 hash 时已扣 1 轮；hysteresis 降格场景下
        丢一轮可接受（本来就要降格）。
        """
        if not self._tracker.is_upgraded(messages, provider=params.provider_name):
            return False

        current_score = compute_complexity_score(messages).score
        if current_score < params.downgrade_thr:
            self._tracker.clear(messages, provider=params.provider_name)
            logger.info(
                "升格主动撤销: score=%.2f < downgrade_thr=%.2f → 切回 %s",
                current_score, params.downgrade_thr, params.flash_model,
            )
            return False  # hysteresis 降格：让后续 step 重新决定

        remaining = self._tracker.remaining(messages, provider=params.provider_name)
        logger.info(
            "持久升格命中 → %s（剩余 %d 轮, score=%.2f）",
            params.pro_model, remaining, current_score,
        )
        body["model"] = params.pro_model
        return True

    def _step_compute_and_commit(
        self, body: Dict[str, Any], messages: List[Dict[str, Any]],
        params: _ProviderParams,
    ) -> None:
        """Steps 3-5：启发式 + Router + throttle 提交。

        三步交织（throttle 在最后基于 did_upgrade 提交），合并到一个方法
        因为 throttle.should_throttle 必须按"本轮是否升格"的最终决定调用，
        分散到 3 个方法会导致状态传递繁琐。
        """
        # Step 3: heuristic 快速路径
        heuristic_result = compute_complexity_score(messages)
        did_upgrade = False
        if heuristic_result.score >= params.heur_thr:
            did_upgrade = True
            logger.info(
                "启发式升格: score=%s >= threshold=%s",
                heuristic_result.score, params.heur_thr,
            )

        # Step 4: BERT/Rule Router 决策
        if not did_upgrade:
            router_score = self._bert.score(messages, body=body)
            if router_score >= params.router_thr:
                logger.info(
                    "Router 升格: score=%.3f >= threshold=%.2f provider=%s "
                    "(heuristic=%.1f/10, user_msgs=%d, user_chars=%d)",
                    router_score, params.router_thr, params.provider_name,
                    heuristic_result.score, heuristic_result.user_msg_count,
                    len(heuristic_result.user_text),
                )
                did_upgrade = True
            else:
                logger.info(
                    "保留 Flash: score=%.3f < threshold=%.2f (heuristic=%.1f/10) → %s",
                    router_score, params.router_thr,
                    heuristic_result.score, params.flash_model,
                )

        # Step 5: throttle 提交 + 升格落地
        # Coding Agent 场景下同一 user 消息连续触发升格 → 强制回退 + 冷却
        if did_upgrade:
            if self._throttle.should_throttle(messages, True):
                did_upgrade = False
                # 同步清持久升格 entry，否则下一轮 Step 2b 会越过 throttle 走 Pro
                self._tracker.clear(messages, provider=params.provider_name)
                logger.info(
                    "升格限流: 连续 %d 次触发 → 强制 Flash（冷却 %d 轮）",
                    self._throttle.max_repeats, self._throttle.cooldown_turns,
                )
        else:
            self._throttle.should_throttle(messages, False)

        if did_upgrade:
            body["model"] = params.pro_model
            self._stash_pending(body, messages, params)
