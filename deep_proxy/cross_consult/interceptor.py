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
from .awareness import build_awareness_prompt
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

    # 追加 system prompt 增量：先 awareness（双家族披露），再原有 tool addendum
    target_name = cc_config.pair_for(source_provider_name) or ""
    awareness_text = ""
    if cc_config.awareness_enabled and target_name:
        awareness_text = build_awareness_prompt(
            source_provider_name=source_provider_name,
            target_provider_name=target_name,
            tool_name=cc_config.tool_name,
            max_calls=cc_config.max_calls_per_request,
        )
    tool_addendum = build_system_prompt_addendum(
        tool_name=cc_config.tool_name,
        max_calls=cc_config.max_calls_per_request,
    )
    addendum = awareness_text + tool_addendum

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
    """从 response.choices[0] 提取所有 name == tool_name 的 tool_calls。

    只处理 choices[0]——DeepProxy 不支持 n>1，与下游 append assistant_msg
    的语义保持一致（避免 tool_call_id 错配进无关 choice）。
    """
    choices = response.get("choices") or []
    if not choices:
        return []
    msg = choices[0].get("message") or {}
    return [
        tc for tc in (msg.get("tool_calls") or [])
        if (tc.get("function") or {}).get("name") == tool_name
    ]


def build_initial_response_from_stream_tool_calls(
    accumulated_tool_calls: list[dict],
) -> dict[str, Any]:
    """把流式累加的 OpenAI tool_calls 包成 execute_cross_consult_loop 期望的
    initial_response 形状（非流式 chat completion 响应模板）。

    供 router.iter_chat_chunks 在 cross_consult 触发时调用——把"流式
    捕获到的 cc tool_call"转译成"伪非流式响应"以便复用现有 loop 入口。
    """
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": accumulated_tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
    }


def synthesize_final_stream_chunk(final_result: dict[str, Any]) -> dict[str, Any]:
    """把 cross_consult 循环完成后的非流式 final_result 还原成一个
    单 chunk 的流式响应（OpenAI streaming chunk 形状）。

    供 router.iter_chat_chunks 在 cc 重发完成后 yield 给客户端——客户端
    继续按 SSE 协议解析；最终响应内容（含 reasoning_content / tool_calls）
    完整保留在这一个合成 chunk 里。
    """
    msg = (final_result.get("choices") or [{}])[0].get("message") or {}
    delta: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
    if msg.get("reasoning_content"):
        delta["reasoning_content"] = msg["reasoning_content"]
        delta["reasoning"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        delta["tool_calls"] = msg["tool_calls"]
    return {
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": "stop",
        }],
    }


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


async def _resolve_consult_tool_call(
    tc: dict,
    *,
    call_count: int,
    target_provider: Provider,
    config: ProxyConfig,
    cc_config: CrossConsultConfig,
) -> tuple[str, bool]:
    """处理单个 cross_consult tool_call：4 路验证 + 必要时调 executor。

    返回 (tool_text, consumed_quota)：
      - tool_text: tool_result 内容（成功结果 或 错误前缀字符串）
      - consumed_quota: True 仅当真正调了 executor（quota / 缺字段 / 超长
        三种验证错误均为 False，不算消耗）

    把 4 路 if/elif/elif/else 抽出，让 execute_cross_consult_loop 主循环
    专注 messages 追加与重发；同时验证逻辑可独立单元测试无需 mock 上游。
    """
    args = _parse_args(tc)
    question = (args.get("question") or "").strip()
    context = args.get("context") or None
    combined_len = len(question) + (len(context) if context else 0)

    # 验证 1：quota 耗尽（最高优先级——已 over budget 时不再 spend）
    if call_count >= cc_config.max_calls_per_request:
        return (
            f"[DeepProxy cross_consult error] quota "
            f"({cc_config.max_calls_per_request}) exhausted for this request",
            False,
        )
    # 验证 2：必填 question 缺失
    if not question:
        return (
            "[DeepProxy cross_consult error] missing required 'question' field",
            False,
        )
    # 验证 3：question+context 超长
    if combined_len > cc_config.max_input_chars:
        return (
            f"[DeepProxy cross_consult error] input too long "
            f"({combined_len} chars > {cc_config.max_input_chars})",
            False,
        )

    # 通过验证 → 实际执行
    tool_text = await execute_consult(
        question=question,
        context=context,
        target_provider=target_provider,
        config=config,
        cc_config=cc_config,
    )
    return tool_text, True


async def execute_cross_consult_loop(
    *,
    body: dict[str, Any],
    initial_response: dict[str, Any],
    source_provider: Provider,
    config: ProxyConfig,
    cc_config: CrossConsultConfig,
    call_litellm_fn,
    process_response_fn=None,  # 可选：每次重发响应过 process_response 再继续
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
            tool_text, consumed = await _resolve_consult_tool_call(
                tc, call_count=call_count,
                target_provider=target_provider, config=config, cc_config=cc_config,
            )
            if consumed:
                call_count += 1
            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": tool_text,
            })

        # 重发原 provider
        response = await call_litellm_fn(config, body, provider=source_provider)
        if process_response_fn is not None:
            response = process_response_fn(response, provider=source_provider)

    # 达到硬上限（防无限循环）——返回最后一次响应
    logger.warning(
        "cross_consult loop reached hard turn limit (%d); returning last response", max_turns
    )
    return response
