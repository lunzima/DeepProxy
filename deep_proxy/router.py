"""核心请求路由器。

统一请求/响应管道（V4 兼容）：

  Chat 端点 → prepare_request（模型名/thinking/参数过滤/推理检查）
            → LiteLLM (acompletion / acompletion stream)
            → process_response（reasoning 兼容字段）

注：FIM 端点已下线，prepare_request 仅服务 chat 请求。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional, TypeVar

import httpx
from litellm.exceptions import RateLimitError, ServiceUnavailableError, APIError

from .compatibility.deepseek_fixes import (
    default_thinking_type,
    is_v4_model,
    normalize_model_name,
    sanitize_stream_options,
)
from .compatibility.error_mapper import (
    map_litellm_error,
    strip_unsupported_params,
)
from .compatibility.reasoning_handler import (
    ReasoningCache,
    StreamingReasoningAccumulator,
    ensure_reasoning_content_persistence,
    process_reasoning_response,
    process_streaming_delta,
    recover_reasoning_content,
)
from .config import ProxyConfig
from .optimization import apply_cheap_optimizations, extract_cot_output, sample_in_range
from .optimization.compressor import SystemPromptCompressor
from .optimization.dynamic_baskets import (
    append_to_system as _append_basket_to_system,
    assemble_paragraph as _assemble_basket_paragraph,
    scenario_from_profile as _scenario_from_profile,
)
from .optimization.silly_priming import (
    pick_one as _pick_silly_priming,
    prepend_to_system as _prepend_silly_to_system,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 仅这些状态码触发重试（5xx + 429）
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _is_retryable_litellm(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, ServiceUnavailableError)):
        return True
    if isinstance(exc, APIError):
        return getattr(exc, "status_code", None) in _RETRYABLE_HTTP
    return False


def _to_litellm_model(model: str) -> str:
    """LiteLLM 需要 `deepseek/<name>` 前缀来路由 provider；HTTP 调用前 SDK 会自动 strip。

    我们内部 (cache / list_models / normalize) 都使用裸名，仅在调 LiteLLM 时附加前缀。
    """
    if not model or "/" in model:
        return model
    return f"deepseek/{model}"


# LiteLLM/DeepSeek 在 message / delta 里会注入这些非标准 OpenAI 字段，
# 严格 Zod 校验（Vercel AI SDK / Cherry Studio）会因此报"类型验证错误"。
_NON_STANDARD_SLOT_FIELDS = ("provider_specific_fields", "audio")
# 这些字段如果是 null 必须省略（OpenAI schema 期望"省略 OR 正常类型"，不接受 null）
_NULL_TO_OMIT_SLOT_FIELDS = ("tool_calls", "function_call", "role", "content")


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


def _clean_response_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """清理非标准/null 字段，避免严格 schema 校验报错。

    LiteLLM 输出常见问题：
    - 顶层 `provider_specific_fields` / `citations` / `service_tier` 等非标准字段
    - `tool_calls: null` / `function_call: null`（schema 要求"省略 OR 数组"）
    - `audio: null` / `audio: {...}`（OpenAI 新字段；旧 AI SDK 不识别，整体拒收）
    - `provider_specific_fields` 在 message 与 choice 两层都会出现（非标准）
    - 流式 delta 偶尔出现 `role: null` / `content: null`
    """
    if not isinstance(payload, dict):
        return payload
    # 顶层非标准字段一律去掉
    for k in ("provider_specific_fields", "citations", "service_tier"):
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
    return payload


# DeepSeek V4 官方规格（DeepSeek API 文档，2026-04 起）：
# - 上下文长度: 1M tokens
# - 输出长度:   最大 384K tokens
_V4_CONTEXT_WINDOW = 1_000_000
_V4_MAX_OUTPUT = 384_000


def _normalize_model_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """补全 OpenAI Model schema 字段，并附加少量"可能有用"的上下文标注。

    OpenAI 官方 /v1/models schema 只有 `id` / `object` / `created` / `owned_by`；
    部分旧版 SDK 还要 legacy 字段 `root` / `parent` / `permission`，缺则拒收。

    上下文长度字段策略（外部查证结论，2026-04）：
    - OpenAI Python SDK：extra="allow"，自定义字段进 .model_extra 可读 → 有用
    - LiteLLM proxy：忽略上游 model_info，只读静态表 + config.yaml → 无用
    - Vercel AI SDK / Cline / Continue / Cursor / Open WebUI：依赖本地表或 UI 手填 → 无用
    - OpenRouter top_provider / Ollama details：仅特定生态识别 → 无用
    故仅保留：(a) 扁平 context_window / max_input_tokens / max_output_tokens
    （OpenAI SDK 可经 model_extra 读到），(b) description 兜底字符串（UI 直接展示）。
    """
    model_id = entry["id"]
    created = int(entry.get("created") or 1700000000)
    out: Dict[str, Any] = {
        "id": model_id,
        "object": entry.get("object", "model"),
        "created": created,
        "owned_by": entry.get("owned_by", "deepseek"),
        # OpenAI legacy
        "root": model_id,
        "parent": None,
        "permission": [
            {
                "id": f"modelperm-{model_id}",
                "object": "model_permission",
                "created": created,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False,
            }
        ],
        # 上下文长度（OpenAI SDK 通过 model_extra 可读；其他客户端按需）
        "context_window": _V4_CONTEXT_WINDOW,
        "max_input_tokens": _V4_CONTEXT_WINDOW,
        "max_output_tokens": _V4_MAX_OUTPUT,
        # 兜底：任何把 description 展示给用户的 UI 都能看到 1M
        "description": "DeepSeek V4 — 1M context window, up to 384K output tokens",
    }
    # 上游已经返回了真值则尊重之
    for k in ("context_window", "max_input_tokens", "max_output_tokens"):
        v = entry.get(k)
        if isinstance(v, int) and v > 0:
            out[k] = v
    return out


def _strip_api_version(base: str) -> str:
    base = base.rstrip("/")
    for suffix in ("/v1", "/beta"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _to_litellm_api_base(api_base: str) -> str:
    """LiteLLM 的 deepseek provider 会做 `api_base + "/chat/completions"`；
    若 api_base 没有 `/v1` 或 `/beta` 后缀，最终 URL 会落在不存在的根路径上
    （governor 返回 401）。这里补齐到 `/v1`。
    """
    if not api_base:
        return api_base
    base = _strip_api_version(api_base)
    return f"{base}/v1"


async def _retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    backoff_base: float,
    is_retryable: Callable[[Exception], bool],
    label: str,
) -> T:
    """通用指数退避重试。第 i 次重试等待 base*(2**i) ± 25% 抖动。"""
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            if attempt >= max_retries or not is_retryable(e):
                raise
            delay = backoff_base * (2 ** attempt)
            delay *= 1.0 + random.uniform(-0.25, 0.25)
            logger.warning("[%s] 第 %d 次重试，错误: %s，等待 %.2fs", label, attempt + 1, e, delay)
            await asyncio.sleep(delay)
            attempt += 1


class DeepProxyRouter:
    """DeepProxy 核心路由器，处理完整的请求/响应生命周期。"""

    def __init__(self, config: ProxyConfig):
        self.config = config
        # 服务端缓存上一轮 reasoning_content；下一轮请求若客户端没回传则补齐
        self._reasoning_cache = ReasoningCache(max_size=1024)
        self._http_client: Optional[httpx.AsyncClient] = None
        # LLM-based system prompt 压缩器（持久化磁盘缓存）
        # 复用 PreciseSamplingConfig 的采样预设：高确定性 + 微抖动，最适合
        # 同义改写类任务（确定性是主要诉求，微随机仅供并行重试）
        self._compressor: Optional[SystemPromptCompressor] = None
        if config.optimization.enabled and config.optimization.compress_skills:
            from pathlib import Path
            self._compressor = SystemPromptCompressor(
                cache_path=Path(config.optimization.compressor_cache_path),
                api_key=config.deepseek.api_key,
                api_base=_to_litellm_api_base(config.deepseek.api_base),
                model=config.optimization.compressor_model,
                sampling=config.precise_sampling,
            )

    # ------------------------------------------------------------------
    # 预处理管道（公共方法，由 main.py 端点层调用）
    # ------------------------------------------------------------------

    async def prepare_request(
        self,
        body: Dict[str, Any],
        *,
        sampling_profile: Any = None,
    ) -> Dict[str, Any]:
        """聊天补全请求预处理管道。

        Args:
            sampling_profile: 若提供（PreciseSamplingConfig / CreativeSamplingConfig
                duck-typed），则强制覆盖 body 中的 4 个采样参数（不是 setdefault）。
                None 时退回旧的 creative_sampling.enabled-based 默认行为，便于测试。
        """
        raw_model = body.get("model", "")

        # 0a. 模型别名隐含特定 thinking 模式（仅当客户端未显式提供 thinking 时应用）：
        #     - deepseek-chat → disabled；deepseek-reasoner → enabled
        #     - claude-* → enabled（OpenAI 兼容端点收到 claude 名字时同样开 reasoning）
        if "thinking" not in body:
            implicit = default_thinking_type(raw_model)
            if implicit is not None:
                body["thinking"] = {"type": implicit}

        # 0b. 模型名称规范化（reasoner/chat 都会被映射到 v4-flash）
        model_routes = [r.model_dump() for r in self.config.model_routes]
        body["model"] = normalize_model_name(raw_model, model_routes)
        model = body.get("model", "")

        # 1. 默认 reasoning_effort=max 注入（仅当未显式 disabled 且未指定）。
        #    官方文档：reasoning_effort 是 thinking 对象的子字段，不是顶层参数。
        if is_v4_model(model):
            thinking = body.get("thinking")
            explicitly_disabled = (
                isinstance(thinking, dict) and thinking.get("type") == "disabled"
            )
            if not explicitly_disabled:
                if not isinstance(thinking, dict):
                    thinking = {}
                    body["thinking"] = thinking
                thinking.setdefault("reasoning_effort", "max")

        # 2. 采样参数：
        #    - 若传入 sampling_profile（生产路径，端口绑定）：强制覆盖客户端值
        #    - 否则（测试 / 单端口）：legacy default 行为（setdefault）
        if sampling_profile is not None:
            sp = sampling_profile
            body["temperature"] = sample_in_range(sp.temperature_min, sp.temperature_max)
            body["top_p"] = sample_in_range(sp.top_p_min, sp.top_p_max)
            body["presence_penalty"] = sample_in_range(
                sp.presence_penalty_min, sp.presence_penalty_max
            )
            body["frequency_penalty"] = sample_in_range(
                sp.frequency_penalty_min, sp.frequency_penalty_max
            )
        elif self.config.creative_sampling.enabled:
            rp = self.config.creative_sampling
            body.setdefault("temperature", sample_in_range(rp.temperature_min, rp.temperature_max))
            body.setdefault("top_p", sample_in_range(rp.top_p_min, rp.top_p_max))
            body.setdefault("presence_penalty",
                            sample_in_range(rp.presence_penalty_min, rp.presence_penalty_max))
            body.setdefault("frequency_penalty",
                            sample_in_range(rp.frequency_penalty_min, rp.frequency_penalty_max))
        else:
            body.setdefault("temperature", 0.6)
            body.setdefault("top_p", 0.95)

        # 3. 参数过滤 — 移除 DeepSeek 不支持的参数（仅 functions / user）
        if self.config.deepseek.strip_unsupported_params:
            body = strip_unsupported_params(body)

        # 4. V4 多轮：用服务端缓存（按对话前缀）补齐 reasoning_content；
        #    补不齐则注入 dummy 占位（保持 thinking=enabled，保留本轮推理能力）
        if is_v4_model(model):
            messages = body.get("messages", [])
            if messages:
                body = ensure_reasoning_content_persistence(
                    messages, body, cache=self._reasoning_cache,
                )

        # 5. 清理空 stream_options
        body = sanitize_stream_options(body)

        # 6. 廉价提示词优化 + 内置 skills（in-process，0 额外上游调用）
        if self.config.optimization.enabled:
            opt = self.config.optimization
            await apply_cheap_optimizations(
                body,
                # A. 通用风格
                avoid_negative_style=opt.avoid_negative_style,
                assume_good_intent=opt.assume_good_intent,
                instruction_priority=opt.instruction_priority,
                independent_analysis=opt.independent_analysis,
                reason_genuinely=opt.reason_genuinely,
                inject_date=opt.inject_date,
                cot_reset=opt.cot_reset,
                # B. 求证 / 反幻觉
                show_math_steps=opt.show_math_steps,
                prefer_multiple_sources=opt.prefer_multiple_sources,
                avoid_fabricated_citations=opt.avoid_fabricated_citations,
                # C. 上下文相关
                json_mode_hint=opt.json_mode_hint,
                safe_inlined_content=opt.safe_inlined_content,
                # D. 消息转换
                re2=opt.re2,
                cot_reflection=opt.cot_reflection,
                readurls=opt.readurls,
                # LLM 压缩器（首次慢、后续命中缓存秒返回）
                compressor=self._compressor,
                http_client=self._get_http_client(),
            )

        # 7. 动态短段注入（场景化 PUA-substance 提示词）
        #    必须在 apply_cheap_optimizations（含 LLM 压缩）之后执行，避免随机
        #    句子进入压缩缓存键、每请求刷新缓存。
        if (
            self.config.optimization.enabled
            and self.config.optimization.dynamic_baskets
            and not body.get("tools")
            and not body.get("tool_choice")
        ):
            scenario = _scenario_from_profile(sampling_profile)
            if scenario:
                paragraph = _assemble_basket_paragraph(
                    scenario,
                    writing_kind=self.config.optimization.writing_basket_kind,
                )
                if paragraph:
                    messages = body.get("messages")
                    if isinstance(messages, list) and messages:
                        _append_basket_to_system(messages, paragraph)

        # 8. 无厘头 expert priming（最后一步）
        #    Always 全场景生效；不进压缩缓存键；插入到 system 消息最前面
        if (
            self.config.optimization.enabled
            and self.config.optimization.silly_expert_priming
            and not body.get("tools")
            and not body.get("tool_choice")
        ):
            priming = _pick_silly_priming()
            if priming:
                messages = body.get("messages")
                if isinstance(messages, list) and messages:
                    _prepend_silly_to_system(messages, priming)

        logger.debug(
            "准备请求: model=%s, stream=%s, params_keys=%s",
            body.get("model"),
            body.get("stream", False),
            list(body.keys()),
        )
        return body

    # ------------------------------------------------------------------
    # LiteLLM 调用（统一入口）
    # ------------------------------------------------------------------

    async def call_litellm(self, body: Dict[str, Any]) -> Dict[str, Any]:
        import litellm

        call_body = dict(body)
        # 移除内部 sentinel 字段（_deepproxy_*），不能泄漏给 LiteLLM
        for k in [k for k in call_body if k.startswith("_deepproxy_")]:
            call_body.pop(k)
        call_body["messages"] = _ensure_string_content(call_body.get("messages", []))
        call_body["model"] = _to_litellm_model(call_body.get("model", ""))
        # 注：必须以 kwarg 形式传递 api_base —— LiteLLM 的 deepseek provider
        # 忽略全局 `litellm.api_base`，无 kwarg 时回退到硬编码 `/beta`，导致
        # URL 错位（governor 401）。
        if self.config.deepseek.api_key:
            call_body["api_key"] = self.config.deepseek.api_key
        if self.config.deepseek.api_base:
            call_body["api_base"] = _to_litellm_api_base(self.config.deepseek.api_base)

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
            return await _retry_async(
                _do,
                max_retries=self.config.deepseek.max_retries,
                backoff_base=self.config.deepseek.retry_backoff_base,
                is_retryable=_is_retryable_litellm,
                label="litellm",
            )
        except Exception as e:
            logger.error("LiteLLM 调用失败: %s", str(e))
            raise map_litellm_error(e) from e

    async def iter_litellm_chunks(
        self,
        body: Dict[str, Any],
        *,
        _accumulator: "StreamingReasoningAccumulator | None" = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """业务层流式产出 dict 流。

        每个 yield 的元素是 OpenAI 风格的 chunk dict（含 choices / usage），
        或 {"error": {...}} 错误终止 dict。生成器自然结束 = 流正常完成。

        协议层（SSE 序列化、`data: [DONE]` 前哨）由调用方负责。
        """
        import litellm

        body = dict(body)
        # 移除内部 sentinel 字段（_deepproxy_*），不能泄漏给 LiteLLM
        for k in [k for k in body if k.startswith("_deepproxy_")]:
            body.pop(k)
        body["stream"] = True
        body["messages"] = _ensure_string_content(body.get("messages", []))
        body["model"] = _to_litellm_model(body.get("model", ""))
        if self.config.deepseek.api_key:
            body["api_key"] = self.config.deepseek.api_key
        if self.config.deepseek.api_base:
            body["api_base"] = _to_litellm_api_base(self.config.deepseek.api_base)

        # 连接建立期可重试（尚未开始向客户端 yield 任何 chunk）
        async def _open() -> Any:
            return await litellm.acompletion(**body)

        try:
            response = await _retry_async(
                _open,
                max_retries=self.config.deepseek.max_retries,
                backoff_base=self.config.deepseek.retry_backoff_base,
                is_retryable=_is_retryable_litellm,
                label="litellm-stream-open",
            )
        except Exception as e:
            logger.error("LiteLLM 流式请求失败（连接建立期）: %s", str(e))
            yield {"error": map_litellm_error(e).detail.get("error", {"message": str(e)})}
            return

        enable_reasoning = self.config.deepseek.enable_reasoning
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
            yield {"error": map_litellm_error(e).detail.get("error", {"message": str(e)})}
            return

    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.deepseek.enable_reasoning:
            response = process_reasoning_response(response)
        return response

    # ------------------------------------------------------------------
    # 端点方法（轻量封装，供 main.py 调用）
    # ------------------------------------------------------------------

    async def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request_messages = list(body.get("messages") or [])
        # 是否需要剥离 CoT Reflection 标签（由 apply_cheap_optimizations 在 prepare_request 时打的标）
        strip_cot = bool(body.pop("_deepproxy_strip_cot", False))
        raw = await self.call_litellm(body)
        result = self.process_response(raw)
        if strip_cot:
            for choice in result.get("choices", []):
                msg = choice.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    msg["content"] = extract_cot_output(msg["content"])
        # 按对话前缀写缓存，供下一轮补齐
        self._reasoning_cache.remember_response(request_messages, result)
        return result

    async def iter_chat_chunks(
        self, body: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """业务层流式 chunk 流（dict 形态）。

        - 每个 yield：OpenAI 风格的 chunk dict（带 reasoning 字段已自愈/累加）
          或 `{"error": {...}}` 错误终止
        - 自然结束 = 流正常完成
        - 期间累加 content / reasoning_content / tool_calls，结束后写 ReasoningCache

        SSE 序列化（`data:` 前缀、`[DONE]` 前哨）由调用方在协议层完成。
        """
        body.pop("_deepproxy_strip_cot", None)
        request_messages = list(body.get("messages") or [])
        accumulator = StreamingReasoningAccumulator(request_messages=request_messages)
        try:
            async for chunk_dict in self.iter_litellm_chunks(body, _accumulator=accumulator):
                yield chunk_dict
        finally:
            accumulator.flush_to_cache(self._reasoning_cache)

    async def chat_completions_stream(
        self, body: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """OpenAI 协议层流式输出：iter_chat_chunks → SSE 字符串。

        负责协议细节：dict → `data: {...}\\n\\n`、错误帧序列化、`data: [DONE]\\n\\n` 前哨。
        """
        async for item in self.iter_chat_chunks(body):
            yield f"data: {__import__('json').dumps(item)}\n\n"
            if "error" in item and "choices" not in item:
                yield "data: [DONE]\n\n"
                return
        yield "data: [DONE]\n\n"

    async def list_models(self) -> Dict[str, Any]:
        """列出可用模型。

        优先从 DeepSeek 上游 `GET /v1/models` 拉取真实清单；上游不可用时退化到
        内置 V4 模型列表。`expose_legacy_models=True` 会附加老别名；
        `model_routes` 中的自定义对外名也会合并进去（去重）。

        OpenAI Model schema 要求 `id` / `object` / `created` / `owned_by` 全部存在
        （DeepSeek 上游返回的条目缺 `created`，会被 OpenAI SDK / AI SDK 严格校验拒绝）。
        本方法保证每条都有这四个字段。
        """
        from .compatibility.deepseek_fixes import DEEPSEEK_MODELS, V4_MODELS

        raw = await self._fetch_upstream_models()
        if not raw:
            raw = list(V4_MODELS.values())

        models = [_normalize_model_entry(m) for m in raw if isinstance(m, dict) and m.get("id")]
        seen = {m["id"] for m in models}

        if self.config.deepseek.expose_legacy_models:
            for m in DEEPSEEK_MODELS.values():
                if m["id"] not in seen:
                    models.append(_normalize_model_entry(m))
                    seen.add(m["id"])

        for route in self.config.model_routes:
            if route.model_name in seen:
                continue
            models.append(_normalize_model_entry({
                "id": route.model_name,
                "owned_by": "deepseek",
            }))
            seen.add(route.model_name)

        return {"object": "list", "data": models}

    async def _fetch_upstream_models(self) -> list:
        api_key = self.config.deepseek.api_key
        if not api_key:
            return []

        base = _strip_api_version(self.config.deepseek.api_base)
        url = f"{base}/v1/models"

        client = self._get_http_client()

        try:
            resp = await client.get(
                url, headers={"authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            return list(data) if isinstance(data, list) else []
        except Exception as e:
            logger.warning("上游 /v1/models 拉取失败，使用本地兜底: %s", e)
            return []

    def _get_http_client(self) -> httpx.AsyncClient:
        """共享的 httpx 客户端，被上游 /v1/models 拉取与 readurls 优化复用。"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0), follow_redirects=True
            )
        return self._http_client

    async def close(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
