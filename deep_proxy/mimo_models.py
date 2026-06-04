"""MiMo 真实模型列表（self-contained，不 import deepseek_models）。

仅涵盖 chat 模型；MiMo 还有 omni / TTS 等模型，未在 DeepProxy 服务范围内。
"""
from __future__ import annotations

from typing import Dict

MIMO_FLASH = "mimo-v2.5"
MIMO_PRO = "mimo-v2.5-pro"

# 真实模型（chat 用途）
MIMO_MODELS: Dict[str, Dict] = {
    MIMO_FLASH: {
        "id": MIMO_FLASH,
        "object": "model",
        "owned_by": "xiaomi",
        "created": 1779840000,  # 2026-05-27 价格更新公告日；首发日期未公开，用公告日占位
    },
    MIMO_PRO: {
        "id": MIMO_PRO,
        "object": "model",
        "owned_by": "xiaomi",
        "created": 1779840000,  # 2026-05-27（同上）
    },
}
