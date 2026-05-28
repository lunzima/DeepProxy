"""MiMo 模型定价表（独立模块）。

数据来源：https://platform.xiaomimimo.com/docs/zh-CN/news/v2.5-price-update
（2026-05-27 价格更新）。数值以 USD per token 表达，与 OpenAI 计价惯例一致。

定价（CNY 折算 USD，按 7.2 汇率简化；后续可改为读取实时汇率）：
  mimo-v2.5      input ¥0.56/Mtok output ¥28/Mtok → USD ≈ 0.078/Mtok / 3.89/Mtok
  mimo-v2.5-pro  input ¥1.40/Mtok output ¥42/Mtok → USD ≈ 0.194/Mtok / 5.83/Mtok

注：缓存命中价格 / 长上下文区段差价未在此表展示（LiteLLM model_cost 不支持分档），
取最常见档（≤256K）价格。
"""

from __future__ import annotations

from typing import Dict

# CNY → USD 简化折算（按 7.2 汇率）
_CNY_TO_USD = 1.0 / 7.2

_MIMO_PRICING: Dict[str, Dict[str, float]] = {
    "mimo-v2.5": {
        "prompt": 0.56 * _CNY_TO_USD,        # USD per Mtok
        "completion": 28.0 * _CNY_TO_USD,
    },
    "mimo-v2.5-pro": {
        "prompt": 1.40 * _CNY_TO_USD,
        "completion": 42.0 * _CNY_TO_USD,
    },
}

_MIMO_CONTEXT_WINDOW = 1_000_000
_MIMO_MAX_OUTPUT = 128_000


def model_pricing(model_id: str) -> Dict[str, float]:
    """返回模型 ID 对应的定价 dict（USD per Mtok）；未知模型返回 {}。"""
    return _MIMO_PRICING.get(model_id, {}).copy()
