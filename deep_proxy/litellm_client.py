"""LiteLLM SDK 调用封装 —— 非流式 / 流式统一入口。

职责范围：
  - 组装 LiteLLM 需要的参数（model 前缀、api_key、api_base）
  - 请求体清理（string content、sentinel 字段）
  - 响应体清理（非标准 / null 字段）
  - 指数退避重试（5xx / 429）
  - 错误映射（LiteLLM 异常 → OpenAI 兼容格式）

复用注意：
  - cross_consult 在响应路径的"非流式重发循环"中复用 call_litellm；
    _assemble_litellm_body 显式覆盖 stream=False 防止 sentinel 泄露成流式调用。

不属于此模块：
  - 请求预处理管道（prepare_request）— 在 router.py
  - reasoning_content 逻辑 — 在 compatibility/reasoning_handler.py
  - 模型列表生成 — 在 models_list.py
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List

from litellm.exceptions import RateLimitError, ServiceUnavailableError, APIError

from .compatibility.reasoning_handler import (
    StreamingReasoningAccumulator,
    process_streaming_delta,
    recover_reasoning_content,
)
from .compatibility.error_mapper import litellm_error_to_dict, map_litellm_error
from .config import ProxyConfig
from .providers import Provider
from .utils import retry_async, strip_api_version

logger = logging.getLogger(__name__)

# 仅这些状态码触发重试（5xx + 429）
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}

# LiteLLM 在 message / delta 里会注入这些非标准 OpenAI 字段（DeepSeek / MiMo
# 均常见），严格 Zod 校验（Vercel AI SDK / Cherry Studio）会因此报"类型验证错误"。
_NON_STANDARD_SLOT_FIELDS = ("provider_specific_fields", "audio")
_NON_STANDARD_TOP_FIELDS = ("provider_specific_fields", "citations", "service_tier")
# 这些字段如果是 null 必须省略（OpenAI schema 要求"省略 OR 正常类型"，不接受 null）
_NULL_TO_OMIT_SLOT_FIELDS = ("tool_calls", "function_call", "role", "content")


def _to_litellm_api_base(api_base: str, *, already_versioned: bool = False) -> str:
    """LiteLLM 的 deepseek provider 会做 `api_base + "/chat/completions"`；
    若 api_base 没有 `/v1` 或 `/beta` 后缀，最终 URL 会落在不存在的根路径上
    （governor 返回 401）。这里补齐到 `/v1`。

    already_versioned=True 时跳过补齐，适用于已显式带 /v1 的 provider api_base。
    """
    if not api_base:
        return api_base
    if already_versioned:
        return api_base
    base = strip_api_version(api_base)
    return f"{base}/v1"


def _to_litellm_model(model: str, *, prefix: str = "deepseek/") -> str:
    """LiteLLM 需要 provider 前缀来路由；HTTP 调用前 SDK 会自动 strip。

    我们内部 (cache / list_models / normalize) 都使用裸名，仅在调 LiteLLM 时附加前缀。
    prefix 默认 "deepseek/"，可按 provider 覆盖（例如 "openai/"）。
    """
    if not model or "/" in model:
        return model
    return f"{prefix}{model}"


def _is_retryable_litellm(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, ServiceUnavailableError)):
        return True
    if isinstance(exc, APIError):
        return getattr(exc, "status_code", None) in _RETRYABLE_HTTP
    return False


# ---------------------------------------------------------------------------
# 请求 / 响应体清理
# ---------------------------------------------------------------------------


def _ensure_string_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把消息中的数组 content 展平为纯字符串。

    DeepSeek API 要求 message.content 必须是字符串（OpenAI 兼容格式允许数组，
    但 DeepSeek 不接受），否则返回 400。
    非文本内容部件（image_url 等）替换为占位符，避免序列化失败。
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            texts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, dict):
                    texts.append(
                        f"[Unsupported content type: {part.get('type', 'unknown')}]"
                    )
            out.append({**msg, "content": "\n\n".join(texts)})
        else:
            out.append(msg)
    return out


def _clean_response_payload(payload: Dict[str, Any]) -> None:
    """原地清理非标准/null 字段，避免严格 schema 校验报错。

    LiteLLM 输出常见问题：
    - 顶层 `provider_specific_fields` / `citations` / `service_tier` 等非标准字段
    - `tool_calls: null` / `function_call: null`（schema 要求"省略 OR 数组"）
    - `audio: null` / `audio: {...}`（OpenAI 新字段；旧 AI SDK 不识别，整体拒收）
    - `provider_specific_fields` 在 message 与 choice 两层都会出现（非标准）
    - 流式 delta 偶尔出现 `role: null` / `content: null`
    """
    if not isinstance(payload, dict):
        return
    # 顶层非标准字段一律去掉
    for k in _NON_STANDARD_TOP_FIELDS:
        payload.pop(k, None)
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        choice.pop("provider_specific_fields", None)
        for slot_key in ("message", "delta"):
            slot = choice.get(slot_key)
            if not isinstance(slot, dict):
                continue
            for k in _NON_STANDARD_SLOT_FIELDS:
                slot.pop(k, None)
            for k in _NULL_TO_OMIT_SLOT_FIELDS:
                if k in slot and slot[k] is None:
                    # 非流式 message.content=null 是合法的（拒绝时模型返回空），保留
                    if slot_key == "message" and k == "content":
                        continue
                    slot.pop(k, None)


# ---------------------------------------------------------------------------
# LiteLLM 调用入口
# ---------------------------------------------------------------------------


def _strip_sentinels(body: Dict[str, Any]) -> Dict[str, Any]:
    """复制 body 并移除内部 _deepproxy_* sentinel 字段，不修改原 body。"""
    call_body = dict(body)
    for k in [k for k in call_body if k.startswith("_deepproxy_")]:
        call_body.pop(k)
    return call_body


def _assemble_litellm_body(
    body: Dict[str, Any],
    config: ProxyConfig,
    *,
    stream: bool = False,
    provider: Provider | None = None,
) -> Dict[str, Any]:
    """从业务 body 组装 LiteLLM 调用参数（共享于流式/非流式路径）。

    provider 给定时按该 provider 的 api_base/api_key/prefix 路由；
    provider=None 时退回 config.deepseek.*（向后兼容老调用点与未传 provider 的测试路径）。
    """
    call_body = _strip_sentinels(body)
    # 显式覆盖：调用方 stream 参数为真理；防止原 body 中残留的 stream=True
    # 在非流式 re-call（如 cross_consult 重发循环）中泄漏，导致 LiteLLM 返回
    # async iterator 而非 dict。
    call_body["stream"] = stream
    call_body["messages"] = _ensure_string_content(call_body.get("messages", []))

    if provider is not None:
        call_body["model"] = _to_litellm_model(
            call_body.get("model", ""), prefix=provider.litellm_prefix,
        )
        if provider.api_key:
            call_body["api_key"] = provider.api_key
        if provider.api_base:
            # api_base 已显式带 /v1 或 /beta 时跳过补齐
            already = provider.api_base.rstrip("/").endswith(("/v1", "/beta"))
            call_body["api_base"] = _to_litellm_api_base(
                provider.api_base, already_versioned=already,
            )
        # MiMo / 其它 OpenAI 兼容 provider 接受 OpenAI 没有定义的扩展字段。
        # 直接放在顶层会让 OpenAI Python SDK 报 "unexpected keyword argument"。
        # 走 extra_body 透传：SDK 把 extra_body 内容合并到 HTTP body 而不做 kwarg 校验。
        if provider.allowed_extra_params:
            extra_body = dict(call_body.get("extra_body") or {})
            for p in provider.allowed_extra_params:
                if p in call_body:
                    extra_body[p] = call_body.pop(p)
            if extra_body:
                call_body["extra_body"] = extra_body
    else:
        # 注：必须以 kwarg 形式传递 api_base —— LiteLLM 的 deepseek provider
        # 忽略全局 `litellm.api_base`，无 kwarg 时回退到硬编码 `/beta`，导致
        # URL 错位（governor 401）。
        call_body["model"] = _to_litellm_model(call_body.get("model", ""))
        if config.deepseek.api_key:
            call_body["api_key"] = config.deepseek.api_key
        if config.deepseek.api_base:
            call_body["api_base"] = _to_litellm_api_base(config.deepseek.api_base)
    return call_body


def _build_error_dict(e: Exception) -> dict:
    """将异常映射为 OpenAI 风格错误 dict（不构造 HTTPException）。"""
    return litellm_error_to_dict(e)[1]


async def call_litellm(
    config: ProxyConfig,
    body: Dict[str, Any],
    *,
    provider: Provider | None = None,
) -> Dict[str, Any]:
    """非流式 LiteLLM 调用 + 重试 + 响应清理。"""
    import litellm

    call_body = _assemble_litellm_body(body, config, provider=provider)

    async def _do() -> Any:
        response = await litellm.acompletion(**call_body)
        dumped = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        # LiteLLM 某些版本的 model_dump() 会丢 reasoning_content，
        # 从原始响应对象兜底恢复。
        recover_reasoning_content(dumped, response)
        # 移除非标准/null 字段，避免严格 schema 客户端拒收
        _clean_response_payload(dumped)
        return dumped

    try:
        return await retry_async(
            _do,
            max_retries=config.deepseek.max_retries,
            backoff_base=config.deepseek.retry_backoff_base,
            is_retryable=_is_retryable_litellm,
            label="litellm",
        )
    except Exception as e:
        logger.error("LiteLLM 调用失败: %s", str(e))
        raise map_litellm_error(e) from e


async def iter_litellm_chunks(
    config: ProxyConfig,
    body: Dict[str, Any],
    *,
    _accumulator: StreamingReasoningAccumulator | None = None,
    provider: Provider | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """业务层流式产出 dict 流。

    每个 yield 的元素是 OpenAI 风格的 chunk dict（含 choices / usage），
    或 {"error": {...}} 错误终止 dict。生成器自然结束 = 流正常完成。

    协议层（SSE 序列化、`data: [DONE]` 前哨）由调用方负责。
    """
    import litellm

    call_body = _assemble_litellm_body(body, config, stream=True, provider=provider)

    # 连接建立期可重试（尚未开始向客户端 yield 任何 chunk）
    async def _open() -> Any:
        return await litellm.acompletion(**call_body)

    try:
        response = await retry_async(
            _open,
            max_retries=config.deepseek.max_retries,
            backoff_base=config.deepseek.retry_backoff_base,
            is_retryable=_is_retryable_litellm,
            label="litellm-stream-open",
        )
    except Exception as e:
        logger.error("LiteLLM 流式请求失败（连接建立期）: %s", str(e))
        yield {"error": _build_error_dict(e)}
        return

    # provider 给定时按 provider.has_reasoning_content；否则 fallback 到老 config 字段
    enable_reasoning = (
        provider.has_reasoning_content if provider is not None
        else config.deepseek.enable_reasoning
    )
    try:
        async for chunk in response:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)

            if enable_reasoning:
                recover_reasoning_content(chunk_dict, chunk)
                for choice in chunk_dict.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta:
                        process_streaming_delta(delta)

            if _accumulator is not None:
                _accumulator.consume(chunk_dict)

            _clean_response_payload(chunk_dict)
            yield chunk_dict
    except Exception as e:
        logger.error("LiteLLM 流式请求中途异常: %s", str(e))
        yield {"error": _build_error_dict(e)}
        return
