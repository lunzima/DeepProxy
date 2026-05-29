"""Cross-Consult executor：单次目标 provider 调用（context-free + 不带 tools）。

调用语义：
- 系统提示词来自 cc_config.consult_system_prompt（短、明确"外部顾问"角色）
- user 消息 = question + 可选 context
- target_provider.pro_model
- 不带 tools / tool_choice（防递归）
- _deepproxy_cross_consult_internal=True sentinel（防递归注入）
- 全程走流式：上游用 iter_litellm_chunks 拉流，按 chunk 心跳累加。
  call_timeout_seconds 语义改为"chunk-间最大空闲"——只要持续有 token / reasoning 流
  到达就继续等下去；连续空闲超过该秒数才视为 hang。深度思考不再被墙钟误杀。
- max_tokens 来自 cc_config
- 失败/超时：返回带前缀的错误字符串，不抛异常（让上层把错误以 tool_result 形式返还 agent）
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import ProxyConfig
from ..providers import Provider
from .config import CrossConsultConfig
from .streaming import aggregate_stream_to_response

logger = logging.getLogger(__name__)


_ERROR_PREFIX = "[DeepProxy cross_consult error]"


async def execute_consult(
    *,
    question: str,
    context: str | None,
    target_provider: Provider,
    config: ProxyConfig,
    cc_config: CrossConsultConfig,
) -> str:
    """对 target_provider.pro_model 执行一次 context-free 流式调用，返回纯文本。

    永远不抛异常——错误条件返回带 _ERROR_PREFIX 前缀的字符串，
    供上层封装成 tool_result 内容返还 agent。
    """
    user_content = question
    if context:
        user_content = f"{question}\n\n背景：\n{context}"

    body: dict[str, Any] = {
        "model": target_provider.pro_model,
        "messages": [
            {"role": "system", "content": cc_config.consult_system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": cc_config.max_output_tokens,
        # 注：stream 字段会被 _assemble_litellm_body 覆盖；iter_litellm_chunks 路径
        # 强制 stream=True，这里写不写都不影响。
        "stream": True,
        # 递归防护 sentinel：prepare_request 检测到此标记跳过 cross_consult 注入
        "_deepproxy_cross_consult_internal": True,
    }

    try:
        result = await aggregate_stream_to_response(
            config, body,
            provider=target_provider,
            idle_timeout=float(cc_config.call_timeout_seconds),
            first_chunk_timeout=float(cc_config.first_chunk_timeout_seconds),
        )
    except Exception as e:  # 防御：aggregator 内部已捕获大多数错误，这里保兜底
        logger.warning("cross_consult upstream unexpected error: %s", e)
        return f"{_ERROR_PREFIX} upstream failed: {e}"

    if "_dp_error" in result:
        err = result["_dp_error"]
        if "timeout" in err:
            logger.warning(
                "cross_consult %s source→target=%s pro_model=%s "
                "(idle=%ds first_chunk=%ds)",
                err, target_provider.name, target_provider.pro_model,
                cc_config.call_timeout_seconds, cc_config.first_chunk_timeout_seconds,
            )
            return f"{_ERROR_PREFIX} {err}"
        logger.warning("cross_consult upstream error: %s", err)
        return f"{_ERROR_PREFIX} upstream failed: {err}"

    try:
        choice = result["choices"][0]
        msg = choice["message"]
        text = msg.get("content")
        if not text:
            # MiMo 在某些 prompt 下把全部内容塞进 reasoning_content；兜底取出
            text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return str(text)
    except (KeyError, IndexError, TypeError) as e:
        logger.warning("cross_consult response parse failed: %s; result=%s", e, result)
        return f"{_ERROR_PREFIX} malformed upstream response"
