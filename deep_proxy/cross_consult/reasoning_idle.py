"""reasoning-content 自适应 idle 超时的共享纯逻辑。

client_stream.py（面向客户端真流式）与 streaming.py（内部聚合成 dict）的并发**骨架**
刻意分叉、勿合并（见两模块 docstring）。但"检测到深度思考 token 后把 idle 预算升级到
何值"这条**纯公式**与"chunk 是否含非空 reasoning_content"这个**判定**不属于骨架——
收敛到此模块单一真相，避免两份内联实现各自漂移。
"""
from __future__ import annotations

from typing import Any


def compute_reasoning_idle(base_idle: float, first_chunk_timeout: float | None) -> float:
    """首次出现 reasoning_content 后,mid-stream idle 预算的升级目标值。

    取 base_idle 与 first_chunk_timeout 的较大值：深度思考的 burst 间隙与初始
    prefill 属同一量级，用 first_chunk_timeout 兜底。first_chunk_timeout 为
    None / 0（未配置 / 禁用）时退回 base_idle（即不升格）。
    """
    return max(base_idle, first_chunk_timeout or base_idle)


def chunk_has_reasoning(chunk: dict[str, Any]) -> bool:
    """chunk 的 choices[].delta 中是否含非空 reasoning_content（深度思考 token）。"""
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        if isinstance(delta.get("reasoning_content"), str) and delta["reasoning_content"]:
            return True
    return False
