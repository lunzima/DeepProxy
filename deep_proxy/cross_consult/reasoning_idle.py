"""reasoning-content 检测的共享纯逻辑。

"chunk 是否含非空 reasoning_content（深度思考 token）" 这个判定收敛到此模块单一真相，
供唯一的超时引擎 consume_with_heartbeat 复用（首见 reasoning 后把 idle 升到 reasoning_idle）。
"""
from __future__ import annotations

from typing import Any


def chunk_has_reasoning(chunk: dict[str, Any]) -> bool:
    """chunk 的 choices[].delta 中是否含非空 reasoning_content（深度思考 token）。"""
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        if isinstance(delta.get("reasoning_content"), str) and delta["reasoning_content"]:
            return True
    return False
