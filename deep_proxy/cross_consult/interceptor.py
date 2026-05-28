"""Cross-Consult 请求路径注入 + 响应路径拦截/重发循环。

请求路径：inject_into_request — 加 tool schema + system prompt 增量
响应路径：拦截 + 重发循环将在 Task 6 加入
"""
from __future__ import annotations

import logging
from typing import Any

from .config import CrossConsultConfig
from .schema import build_system_prompt_addendum, build_tool_schema

logger = logging.getLogger(__name__)


def inject_into_request(
    body: dict[str, Any],
    *,
    source_provider_name: str,
    cc_config: CrossConsultConfig,
) -> bool:
    """请求路径注入。返回 True 表示已注入，False 表示跳过。

    跳过条件：
    - body 带 _deepproxy_cross_consult_internal sentinel
    - cc_config.pair_for() 返回 None（disabled 或当前 provider 在 pairs 中无对偶）
    """
    if body.get("_deepproxy_cross_consult_internal"):
        return False
    if cc_config.pair_for(source_provider_name) is None:
        return False

    # 注入工具到 tools 数组（append，不替换用户工具）
    tool_schema = build_tool_schema(tool_name=cc_config.tool_name)
    existing_tools = body.get("tools")
    if isinstance(existing_tools, list):
        existing_tools.append(tool_schema)
    else:
        body["tools"] = [tool_schema]

    # 追加 system prompt 增量
    addendum = build_system_prompt_addendum(
        tool_name=cc_config.tool_name,
        max_calls=cc_config.max_calls_per_request,
    )
    messages = body.get("messages")
    if isinstance(messages, list):
        # 找首条 system；无则新建一条 prepend
        sys_idx = next(
            (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"),
            None,
        )
        if sys_idx is None:
            messages.insert(0, {"role": "system", "content": addendum.lstrip()})
        else:
            existing = messages[sys_idx].get("content", "")
            if isinstance(existing, str):
                messages[sys_idx]["content"] = existing + addendum
            elif isinstance(existing, list):
                # 多模态 system content 列表——追加一条 text 项
                existing.append({"type": "text", "text": addendum.lstrip()})

    logger.debug("cross_consult injected for provider=%s", source_provider_name)
    return True
