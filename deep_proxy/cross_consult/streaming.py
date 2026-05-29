"""流式聚合 helper：消费 iter_litellm_chunks 并把 deltas 拼回非流式响应 dict。

为什么需要这层：cross_consult 的内部往返（executor 调外部 pro 模型 + 重发原 provider）
默认会触发深度思考。墙钟超时（asyncio.wait_for(call_litellm, timeout)）在模型仍在
流式产出 reasoning token 时会被误杀。改走 streaming + 按 chunk 心跳的 idle timeout：
只要 N 秒内仍有 chunk 到达就继续，连续 idle_timeout 秒没有 chunk 才视为 hang。

输出形状刻意贴近非流式 chat.completion 响应，让现有的 process_response / extract_*
工具链与 cross_consult interceptor 无需感知差异。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Awaitable, Callable

from ..config import ProxyConfig
from ..litellm_client import iter_litellm_chunks
from ..providers import Provider
from ..utils import merge_tool_call_deltas

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
    iter_fn: IterFn | None = None,
) -> dict[str, Any]:
    """流式调上游、按 chunk 累加，返回非流式 chat.completion 风格 dict。

    返回形状：
        {
          "choices": [{
            "message": {"role": "assistant",
                        "content": "...", "reasoning_content": "...",
                        "tool_calls": [...]},
            "finish_reason": "stop" | "tool_calls" | ...,
          }]
        }

    错误（连接失败 / 流中途异常 / 超时）以 {"_dp_error": str} 返回，调用方
    自行决定如何包装（executor 包成错误前缀字符串；interceptor 把它当成空响应处理）。

    两个超时预算（刻意分离，见模块 docstring）：
      - first_chunk_timeout：等待**首个** chunk（prefill / TTFT + 推理预热）的上限。
        大上下文重发的 time-to-first-chunk 远长于 chunk 间隙，故单独给宽预算。
        None / 0 / 负数 = 退回 idle_timeout 守护首 chunk（向后兼容）。
      - idle_timeout：首 chunk 之后，相邻 chunk 间允许的最长无活动时间（真正的
        mid-stream hang tripwire）。0 / 负数 = 不限。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason: str | None = None
    usage: dict | None = None

    # 默认到模块级名（注：通过模块属性查找，便于测试 patch 整个 streaming.iter_litellm_chunks）
    fn = iter_fn if iter_fn is not None else iter_litellm_chunks
    iterator = fn(config, body, provider=provider).__aiter__()
    chunk_count = 0
    start = time.monotonic()
    while True:
        first = chunk_count == 0
        # 首 chunk 用 first_chunk_timeout（未给则退回 idle_timeout）；之后用 idle_timeout
        wait_timeout = (
            first_chunk_timeout
            if first and first_chunk_timeout and first_chunk_timeout > 0
            else idle_timeout
        )
        try:
            if wait_timeout and wait_timeout > 0:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=wait_timeout)
            else:
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            phase = "first_chunk" if first else "mid_stream"
            elapsed = time.monotonic() - start
            logger.warning(
                "aggregate_stream %s timeout after %.1fs (chunks=%d, elapsed=%.1fs)",
                phase, wait_timeout, chunk_count, elapsed,
            )
            return {
                "_dp_error": (
                    f"{phase} timeout after {wait_timeout}s "
                    f"({chunk_count} chunks received)"
                )
            }
        chunk_count += 1

        if isinstance(chunk.get("error"), dict) and not chunk.get("choices"):
            err = chunk["error"]
            msg = err.get("message") if isinstance(err, dict) else None
            return {"_dp_error": msg or str(err)}

        if chunk.get("usage"):
            usage = chunk["usage"]

        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            # reasoning_content 在 iter_litellm_chunks 已被 process_streaming_delta 规整
            r = delta.get("reasoning_content")
            if isinstance(r, str):
                reasoning_parts.append(r)
            tcs = delta.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                tool_calls = merge_tool_call_deltas(tool_calls, tcs)
            fr = ch.get("finish_reason")
            if fr:
                finish_reason = fr

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
    iter_fn: IterFn | None = None,
) -> dict[str, Any]:
    """`call_litellm_fn` 兼容签名的流式封装。

    供 execute_cross_consult_loop 的 call_litellm_fn 参数复用——保持 (config, body,
    *, provider) -> dict 接口不变，但内部走流式，避免在重发时被墙钟超时拦截。
    错误（_dp_error）映射成空 message 让 loop 自然终止，由外层错误监控处理。
    """
    result = await aggregate_stream_to_response(
        config, body, provider=provider,
        idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
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
