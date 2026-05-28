"""flash_upgrade 在多 provider 下的行为测试。"""
from __future__ import annotations

import pytest

from deep_proxy.config import FlashUpgradeConfig


def test_flash_upgrade_per_provider_threshold():
    """per_provider 阈值覆盖默认 router_threshold / heuristic_threshold。"""
    cfg = FlashUpgradeConfig(
        router_threshold=0.60,
        heuristic_threshold=7.5,
        per_provider={
            "mimo": {"router_threshold": 0.65},
        },
    )
    assert cfg.router_threshold == 0.60
    assert cfg.threshold_for_provider("mimo", "router_threshold") == 0.65
    assert cfg.threshold_for_provider("deepseek", "router_threshold") == 0.60
    # heuristic 在 per_provider 未覆盖时 fallback 到默认
    assert cfg.threshold_for_provider("mimo", "heuristic_threshold") == 7.5


async def test_maybe_upgrade_uses_provider_pro_model(
    router_dual, provider_mimo,
):
    """升格时切到 provider.pro_model（不是硬编码 deepseek-v4-pro）。"""
    # 直接构造一个会触发启发式升格的 body（很长的复杂用户消息）
    long_prompt = (
        "请系统性地分析以下需求并设计完整方案：" + "X" * 3000
        + "\n要求：1. 列出所有边界情况 2. 给出每种情况的处理策略 3. 提供测试用例 "
        "4. 评估复杂度 5. 考虑并发安全 6. 考虑向后兼容"
    )
    body = {
        "model": provider_mimo.flash_model,
        "messages": [{"role": "user", "content": long_prompt}],
        "_deepproxy_upgrade": True,  # sentinel-based force upgrade
    }
    await router_dual.prepare_request(
        body,
        sampling_profile=router_dual.config.creative_sampling,
        provider=provider_mimo,
    )
    # sentinel 应将其升格到 mimo-v2.5-pro（而不是 deepseek-v4-pro）
    assert body["model"] == provider_mimo.pro_model, f"unexpected model: {body['model']}"


async def test_upgrade_state_isolated_across_providers(
    router_dual, provider_deepseek, provider_mimo,
):
    """同一对话指纹在 deepseek 升格后，切到 mimo 时不应残留升格状态。"""
    msgs = [{"role": "user", "content": "复杂任务" * 200}]

    # 1) 在 deepseek 名下注入升格状态
    fp, last_h = router_dual._upgrade_tracker.snapshot_keys(msgs)
    router_dual._upgrade_tracker.set_remaining_by_key(
        fp, last_h, 2, provider="deepseek",
    )

    # 2) 切到 mimo 同样的 msgs：升格 tracker 不应命中
    assert not router_dual._upgrade_tracker.is_upgraded(msgs, provider="mimo")
    # deepseek 那边仍命中
    assert router_dual._upgrade_tracker.is_upgraded(msgs, provider="deepseek")
