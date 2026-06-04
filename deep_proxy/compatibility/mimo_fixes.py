"""MiMo OpenAI 兼容端点的协议差异适配。

MiMo 协议与 OpenAI 标准的主要差异：
- `reasoning_effort` 是顶层字段（OpenAI 新约定），不是 thinking 子字段
- `reasoning_effort` 仅接受 low / medium / high，不接受 max
- `thinking.type=disabled` 可禁用思考
- 默认行为：观测到 token-plan-cn 端点 enabled，但文档说 disabled（2026-05 观测，
  可能随上游更新变化）—— 不依赖默认
"""
from __future__ import annotations

from typing import Any

_MIMO_VALID_EFFORTS = frozenset({"low", "medium", "high"})


def inject_top_level_reasoning_effort(body: dict[str, Any], *, value: str) -> None:
    """在请求体顶层注入 reasoning_effort（MiMo OpenAI 兼容端点的约定）。

    跳过条件：
    - thinking.type == "disabled"：用户显式禁用思考，不应该再注入 effort
    - 已存在 reasoning_effort 字段：用户/客户端已显式设置，不覆盖

    MiMo 只接受 low/medium/high；把 "max"（Provider.reasoning_effort_value 的默认值，
    DeepSeek 取向）及任何越界值钳到 "high"，避免 config 漏设时 MiMo 400。
    """
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return
    if "reasoning_effort" in body:
        return
    body["reasoning_effort"] = value if value in _MIMO_VALID_EFFORTS else "high"
