"""核心请求路由器。

统一请求/响应管道（V4 兼容）：

  Chat 端点 → prepare_request（模型名/thinking/参数过滤/推理检查）
            → LiteLLM (acompletion / acompletion stream)
            → process_response（reasoning 兼容字段）

注：FIM 端点已下线，prepare_request 仅服务 chat 请求。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from .compatibility.deepseek_fixes import (
    default_thinking_type,
    is_v4_model,
    normalize_model_name,
    sanitize_stream_options,
)
from .deepseek_models import V4_FLASH, V4_PRO
from .compatibility.error_mapper import (
    strip_unsupported_params,
)
from .compatibility.reasoning_handler import (
    ReasoningCache,
    StreamingReasoningAccumulator,
    ensure_reasoning_content_persistence,
    process_reasoning_response,
)
from .config import ProxyConfig
from .litellm_client import call_litellm, iter_litellm_chunks, _to_litellm_api_base
from .models_list import build_models_list, fetch_upstream_models
from .optimization import apply_cheap_optimizations, extract_cot_output, sample_in_range
from .optimization.compressor import SystemPromptCompressor
from .optimization.dynamic_baskets import (
    append_to_system as _append_basket_to_system,
    assemble_paragraph as _assemble_basket_paragraph,
    scenario_from_profile as _scenario_from_profile,
)
from .optimization.flash_upgrade import (
    DailyUpgradeThrottle,
    UpgradeTracker,
    _flatten_messages,
    compute_complexity_score,
    extra_body_requests_upgrade,
    has_upgrade_sentinel,
)
from .optimization.upgrade_router import create_router
from .optimization.silly_priming import (
    pick_one as _pick_silly_priming,
    prepend_to_system as _prepend_silly_to_system,
)

logger = logging.getLogger(__name__)


class DeepProxyRouter:
    """DeepProxy 核心路由器，处理完整的请求/响应生命周期。"""

    def __init__(self, config: ProxyConfig):
        self.config = config
        # 服务端缓存上一轮 reasoning_content；下一轮请求若客户端没回传则补齐
        self._reasoning_cache = ReasoningCache(max_size=1024)
        self._http_client: Optional[httpx.AsyncClient] = None
        # Flash→Pro 升格跟踪器 + 路由决策器 + 防重复刷屏
        self._upgrade_tracker = UpgradeTracker()
        self._upgrade_router = self._build_upgrade_router()
        self._upgrade_throttle = DailyUpgradeThrottle()
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

        # 0c. Flash→Pro 选择性升格路由（仅 v4-flash + 启用时）
        #     在全部后续处理之前改写 model，让 thinking/sampling/skills 走 Pro 路径。
        if (
            self.config.flash_upgrade.enabled
            and model == V4_FLASH
        ):
            self._maybe_upgrade(body)
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

        # 4. 清理空 stream_options
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

        # 9. V4 多轮 reasoning 自愈：在全部消息修改之后执行，确保
        #    缓存键与 remember_response 存储时的对话前缀一致。
        if is_v4_model(model):
            messages = body.get("messages", [])
            if messages:
                body = ensure_reasoning_content_persistence(
                    messages, body, cache=self._reasoning_cache,
                )

        logger.debug(
            "准备请求: model=%s, stream=%s, params_keys=%s",
            body.get("model"),
            body.get("stream", False),
            list(body.keys()),
        )
        return body

    # ------------------------------------------------------------------
    # Flash→Pro 升格路由（Layer 0–3）
    # ------------------------------------------------------------------

    def _build_upgrade_router(self):
        """初始化升格决策器（Layer 0）。"""
        cfg = self.config.flash_upgrade
        if cfg.router_type == "bert" and cfg.bert_checkpoint:
            return create_router("bert", checkpoint_path=cfg.bert_checkpoint)
        return create_router("rule")

    def _maybe_upgrade(
        self,
        body: Dict[str, Any],
    ) -> None:
        """Flash→Pro 升格路由主逻辑（Layer 0–3，全部 upfront）。

        决策顺序（短路）：
          1. Sentinel 强制升格（最高优先级）
          2. 对话已升格（Layer 3 持久化）
          3. 启发式快速路径（Layer 1）
          4. Router 决策（Layer 0）
        """
        cfg = self.config.flash_upgrade
        messages = body.get("messages", [])

        # ── Step 1: Sentinel / extra_body 强制升格 ──
        if has_upgrade_sentinel(messages) or extra_body_requests_upgrade(body):
            logger.info("Sentinel 强制升格 → %s", V4_PRO)
            body["model"] = V4_PRO
            self._upgrade_tracker.set_remaining(messages, cfg.persist_turns)
            return

        # ── Step 2: 对话已处于升格状态（Layer 3 持久化） ──
        if self._upgrade_tracker.is_upgraded(messages):
            remaining = self._upgrade_tracker.remaining(messages)
            logger.info("持久升格命中 → %s（剩余 %d 轮）", V4_PRO, remaining)
            body["model"] = V4_PRO
            return

        # ── Step 3: 启发式快速路径（Layer 1） ──
        heuristic_score = compute_complexity_score(messages)
        did_upgrade = False
        if heuristic_score >= cfg.heuristic_threshold:
            did_upgrade = True
            logger.info("启发式升格: score=%s >= threshold=%s",
                        heuristic_score, cfg.heuristic_threshold)

        # ── Step 4: Router 决策（Layer 0） ──
        # 注：v4-flash 处理简单编码任务效果已极好，不再对 coding_port 做阈值优惠。
        if not did_upgrade:
            router_score = self._upgrade_router.score(messages, body=body)
            if router_score >= cfg.router_threshold:
                user_text = _flatten_messages(messages, user_only=True)
                user_msg_count = sum(1 for m in messages if m.get("role") == "user")
                logger.info(
                    "Router 升格: score=%.3f >= threshold=%.2f "
                    "(heuristic=%.1f/10, user_msgs=%d, user_chars=%d)",
                    router_score, cfg.router_threshold,
                    heuristic_score, user_msg_count, len(user_text),
                )
                did_upgrade = True
            else:
                logger.info(
                    "保留 Flash: score=%.3f < threshold=%.2f (heuristic=%.1f/10) → %s",
                    router_score, cfg.router_threshold, heuristic_score, V4_FLASH,
                )

        # ── Step 5: 防重复刷屏（Layer 2） ──
        # Coding Agent 场景下同一复杂消息可能重复多次；连续 N 次相同
        # user 消息触发升格后，强制回退到 Flash 并冷却，避免浪费。
        if did_upgrade:
            if self._upgrade_throttle.should_throttle(messages, True):
                did_upgrade = False
                logger.info(
                    "升格限流: 连续 %d 次触发 → 强制 Flash（冷却 %d 轮）",
                    self._upgrade_throttle._max, self._upgrade_throttle._cooldown,
                )
        else:
            self._upgrade_throttle.should_throttle(messages, False)

        if did_upgrade:
            body["model"] = V4_PRO
            self._upgrade_tracker.set_remaining(messages, cfg.persist_turns)

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
        raw = await call_litellm(self.config, body)
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
            async for chunk_dict in iter_litellm_chunks(self.config, body, _accumulator=accumulator):
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
        """列出可用模型（OpenRouter 风格）。

        优先从 DeepSeek 上游 `GET /v1/models` 拉取真实清单；上游不可用时退化到
        内置 V4 模型列表（含 `[1m]` 变体）。`expose_legacy_models=True` 会附加老别名；
        `model_routes` 中的自定义对外名也会合并进去（去重）。
        """
        raw = await fetch_upstream_models(
            self.config.deepseek.api_key,
            self.config.deepseek.api_base,
            self._get_http_client(),
        )
        models = build_models_list(
            raw,
            expose_legacy_models=self.config.deepseek.expose_legacy_models,
            model_routes=[r.model_dump() for r in self.config.model_routes],
        )
        return {"object": "list", "data": models}

    # ------------------------------------------------------------------
    # 预处理管道（公共方法，由 main.py 端点层调用）
    # ------------------------------------------------------------------
