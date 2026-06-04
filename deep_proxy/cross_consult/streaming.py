"""流式聚合 helper：驱动统一超时引擎 consume_with_heartbeat，把 deltas 拼回非流式响应 dict。

为什么需要这层：cross_consult 的内部往返（executor 调外部 pro 模型 + 重发原 provider）
默认会触发深度思考。纯墙钟超时会在模型仍流式产出 reasoning token 时误杀，故复用与客户端
真流式同一个按-chunk idle 引擎（只要 N 秒内仍有 chunk 到达就继续）；本模块只是**忽略心跳、
把 chunk 聚合成 dict** 的薄消费者。

输出形状刻意贴近非流式 chat.completion 响应，让现有 process_response / extract_* 工具链
与 cross_consult interceptor 无需感知差异。

注：本模块仅供**内部聚合**——executor.py 的 consult 调用，以及非流式 chat_completions
路径经 stream_aggregated_call 的重发。面向客户端的**真流式**（逐 token 透传 + 心跳）见
client_stream.py（共享同一 consume_with_heartbeat 引擎）。
"""
from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Callable

from ..config import ProxyConfig
from ..litellm_client import iter_litellm_chunks
from ..providers import Provider
from ..utils import is_error_frame, merge_tool_call_deltas

logger = logging.getLogger(__name__)


# IterFn signature matches iter_litellm_chunks（流式产出 chunk dict 或 {"error": ...}）
IterFn = Callable[..., AsyncGenerator[dict[str, Any], None]]


async def aggregate_stream_to_response(
    config: ProxyConfig,
    body: dict[str, Any],
    *,
    provider: Provider | None,
    idle_timeout: float,
    first_chunk_timeout: float | None = None,
    reasoning_idle: float | None = None,
    heartbeat_seconds: float = 10.0,
    iter_fn: IterFn | None = None,
) -> dict[str, Any]:
    """流式调上游、按 chunk 累加，返回非流式 chat.completion 风格 dict。

    返回形状：
        {"choices": [{"message": {"role": "assistant", "content": ...,
          "reasoning_content": ..., "tool_calls": [...]},
          "finish_reason": "stop" | "tool_calls" | ...}]}

    错误（连接失败 / 流中途异常 / 超时）以 {"_dp_error": str} 返回，调用方自行包装
    （executor 包成错误前缀字符串；interceptor 当成空响应处理）。

    超时与 reasoning 自适应走与客户端真流式**同一个引擎**（consume_with_heartbeat）：
    忽略其心跳哨兵、累加 chunk、_Timeout → _dp_error。first/idle/reasoning_idle 语义与
    client_stream 完全一致（单引擎、单配置）。first_chunk_timeout 未给则退回 idle_timeout。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason: str | None = None
    usage: dict | None = None

    # 延迟导入避免 client_stream → interceptor → executor → streaming 循环
    from .client_stream import consume_with_heartbeat, _HEARTBEAT, _Timeout

    # 默认到模块级名（注：通过模块属性查找，便于测试 patch 整个 streaming.iter_litellm_chunks）
    fn = iter_fn if iter_fn is not None else iter_litellm_chunks
    upstream = fn(config, body, provider=provider)
    gen = consume_with_heartbeat(
        upstream, idle_timeout=idle_timeout,
        # 未给（None）才退回 idle 守护首 chunk；显式 0/负数保留"禁用首 chunk 超时"语义
        first_chunk_timeout=(first_chunk_timeout if first_chunk_timeout is not None
                             else idle_timeout),
        heartbeat_seconds=heartbeat_seconds, log_label="aggregate_stream",
        reasoning_idle=reasoning_idle,
    )
    try:
        async for item in gen:
            if item is _HEARTBEAT:
                continue
            if isinstance(item, _Timeout):
                logger.warning("aggregate_stream %s timeout after %.1fs",
                               item.phase, item.seconds)
                return {"_dp_error": f"{item.phase} timeout after {item.seconds}s"}
            chunk = item
            if is_error_frame(chunk):
                err = chunk["error"]
                msg = err.get("message") if isinstance(err, dict) else None
                return {"_dp_error": msg or str(err)}
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                r = delta.get("reasoning_content")
                if isinstance(r, str):
                    reasoning_parts.append(r)
                tcs = delta.get("tool_calls")
                if isinstance(tcs, list) and tcs:
                    tool_calls = merge_tool_call_deltas(tool_calls, tcs)
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr
    finally:
        await gen.aclose()

    message: dict[str, Any] = {"role": "assistant"}
    content_text = "".join(content_parts)
    # OpenAI 风格：tool_calls 在场时 content 通常为 None；纯文本时给字符串
    if tool_calls:
        message["content"] = content_text or None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = content_text
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)

    response: dict[str, Any] = {
        "choices": [{
            "message": message,
            "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            "index": 0,
        }],
    }
    if usage is not None:
        response["usage"] = usage
    return response


async def stream_aggregated_call(
    config: ProxyConfig,
    body: dict[str, Any],
    *,
    provider: Provider | None = None,
    idle_timeout: float = 30.0,
    first_chunk_timeout: float | None = None,
    reasoning_idle: float | None = None,
    heartbeat_seconds: float = 10.0,
    iter_fn: IterFn | None = None,
) -> dict[str, Any]:
    """`call_litellm_fn` 兼容签名的流式封装。

    供非流式 chat_completions 路径的 execute_cross_consult_loop 作 call_litellm_fn
    复用——保持 (config, body, *, provider) -> dict 接口不变，但内部走流式，避免在
    重发时被墙钟超时拦截。错误（_dp_error）映射成一条 content 为错误前缀串
    （"[DeepProxy cross_consult resend error] ..."）的 assistant message，让 loop
    收到有意义响应后自然终止，由外层错误监控处理。
    """
    result = await aggregate_stream_to_response(
        config, body, provider=provider,
        idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
        reasoning_idle=reasoning_idle, heartbeat_seconds=heartbeat_seconds,
        iter_fn=iter_fn,
    )
    if "_dp_error" in result:
        logger.warning("cross_consult resend streaming failed: %s", result["_dp_error"])
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"[DeepProxy cross_consult resend error] {result['_dp_error']}",
                },
                "finish_reason": "stop",
                "index": 0,
            }],
        }
    return result
