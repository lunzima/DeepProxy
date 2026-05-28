"""Cross-Consult 请求路径注入 + 响应路径拦截/重发循环。

请求路径：inject_into_request — 加 tool schema + system prompt 增量
响应路径：execute_cross_consult_loop — 拦截 + 重发循环（非流式）
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import ProxyConfig
from ..providers import Provider
from .config import CrossConsultConfig
from .executor import execute_consult
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


# ---------------------------------------------------------------------------
# 响应路径拦截 + 重发循环（非流式）
# ---------------------------------------------------------------------------

def _extract_cross_consult_tool_calls(response: dict[str, Any], tool_name: str) -> list[dict]:
    """从 response 提取所有 name == tool_name 的 tool_calls。返回 OpenAI 风格 dict 列表。"""
    out: list[dict] = []
    for choice in response.get("choices") or []:
        msg = choice.get("message") or {}
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if fn.get("name") == tool_name:
                out.append(tc)
    return out


def _parse_args(tc: dict) -> dict:
    raw = (tc.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


async def execute_cross_consult_loop(
    *,
    body: dict[str, Any],
    initial_response: dict[str, Any],
    source_provider: Provider,
    config: ProxyConfig,
    cc_config: CrossConsultConfig,
    call_litellm_fn,
) -> dict[str, Any]:
    """响应路径循环（非流式）。

    1. 检查 initial_response 是否含 cross_consult tool_call
    2. 若无：原样返回
    3. 若有：execute_consult 拿结果 → append assistant tool_call msg + tool_result msg 到 messages
       → re-call original provider → 循环
    4. 计数达到 max_calls_per_request 时，后续 cross_consult tool_call 返回 quota error tool_result
       （但继续循环让 agent 处理 / 返回最终响应）

    防无限循环：硬轮次上限 = max_calls_per_request * 2 + 1，
    确保 quota 耗尽场景下 agent 最多再得到一次重发机会后必定退出。
    """
    if not cc_config.enabled:
        return initial_response

    target_name = cc_config.pair_for(source_provider.name)
    if target_name is None:
        return initial_response

    target_provider = config.providers.get(target_name)
    if target_provider is None:
        return initial_response

    response = initial_response
    call_count = 0
    max_turns = cc_config.max_calls_per_request * 2 + 1

    for _turn in range(max_turns):
        tool_calls = _extract_cross_consult_tool_calls(response, cc_config.tool_name)
        if not tool_calls:
            return response

        # assistant 消息（含 tool_calls）追加到对话历史
        assistant_msg = response["choices"][0]["message"]
        body["messages"].append(assistant_msg)

        for tc in tool_calls:
            args = _parse_args(tc)
            question = (args.get("question") or "").strip()
            context = args.get("context") or None
            combined_len = len(question) + (len(context) if context else 0)

            if call_count >= cc_config.max_calls_per_request:
                tool_text = (
                    f"[DeepProxy cross_consult error] quota "
                    f"({cc_config.max_calls_per_request}) exhausted for this request"
                )
            elif not question:
                tool_text = "[DeepProxy cross_consult error] missing required 'question' field"
            elif combined_len > cc_config.max_input_chars:
                tool_text = (
                    f"[DeepProxy cross_consult error] input too long "
                    f"({combined_len} chars > {cc_config.max_input_chars})"
                )
            else:
                tool_text = await execute_consult(
                    question=question,
                    context=context,
                    target_provider=target_provider,
                    config=config,
                    cc_config=cc_config,
                )
                call_count += 1

            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": tool_text,
            })

        # 重发原 provider
        response = await call_litellm_fn(config, body, provider=source_provider)

    # 达到硬上限（防无限循环）——返回最后一次响应
    logger.warning(
        "cross_consult loop reached hard turn limit (%d); returning last response", max_turns
    )
    return response
