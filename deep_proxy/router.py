"""核心请求路由器。

统一请求/响应管道（V4 兼容）：

  Chat 端点 → prepare_request（模型名/thinking/参数过滤/推理检查）
            → LiteLLM (acompletion / acompletion stream)
            → process_response（reasoning 兼容字段）

注：FIM 端点已下线，prepare_request 仅服务 chat 请求。
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx

from .compatibility.deepseek_fixes import (
    default_thinking_type,
    ensure_thinking_dict,
    has_tools,
    is_thinking_disabled,
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
from .config import ProxyConfig, CreativeSamplingConfig
from .utils import SSE_DONE, append_to_system_message, prepend_to_system_message
from .litellm_client import call_litellm, iter_litellm_chunks, _to_litellm_api_base
from .models_list import build_models_list, fetch_upstream_models
from .optimization import apply_cheap_optimizations, extract_cot_output, sample_in_range
from .optimization.compressor import SystemPromptCompressor
from .optimization.dynamic_baskets import (
    assemble_paragraphs as _assemble_basket_paragraphs,
    scenario_from_profile as _scenario_from_profile,
)
from .optimization.flash_upgrade import (
    RepeatUpgradeThrottle,
    UpgradeTracker,
    compute_complexity_score,
    extra_body_requests_upgrade,
    has_upgrade_sentinel,
)
from .optimization.upgrade_router import create_router
from .optimization.silly_priming import (
    pick_n as _pick_silly_n,
    wrap_for_injection as _wrap_silly_for_injection,
)
from .optimization.think_steering import (
    inject_inner_os_marker,
)

logger = logging.getLogger(__name__)


class DeepProxyRouter:
    """DeepProxy 核心路由器，处理完整的请求/响应生命周期。"""

    def __init__(self, config: ProxyConfig):
        self.config = config
        # 预序列化 model_routes 为 dict 列表，避免每请求重复 model_dump()
        self._model_routes_dicts = [r.model_dump() for r in config.model_routes]
        # 服务端缓存上一轮 reasoning_content；下一轮请求若客户端没回传则补齐
        self._reasoning_cache = ReasoningCache(max_size=1024)
        self._http_client: httpx.AsyncClient | None = None
        # Flash→Pro 升格跟踪器 + 路由决策器 + 防重复刷屏
        self._upgrade_tracker = UpgradeTracker()
        self._upgrade_router = self._build_upgrade_router()
        self._upgrade_throttle = RepeatUpgradeThrottle()
        # LLM-based system prompt 压缩器（持久化磁盘缓存）
        # 复用 PreciseSamplingConfig 的采样预设：高确定性 + 微抖动，最适合
        # 同义改写类任务（确定性是主要诉求，微随机仅供并行重试）
        self._compressor: SystemPromptCompressor | None = None
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
        body: dict[str, Any],
        *,
        sampling_profile: Any = None,
    ) -> dict[str, Any]:
        """聊天补全请求预处理管道。

        Args:
            sampling_profile: 若提供（PreciseSamplingConfig / CreativeSamplingConfig
                duck-typed），则强制覆盖 body 中的 4 个采样参数（不是 setdefault）。
                None 时退回旧的 creative_sampling.enabled-based 默认行为，便于测试。
        """
        raw_model = body.get("model", "")

        # 0a. 模型别名隐含特定 thinking 模式（仅当客户端未显式提供 thinking 时应用）：
        #     政策：除显式 disabled 与 deepseek-chat（"非思考模式"专属别名）外，默认 enabled。
        #     - deepseek-chat → disabled；deepseek-reasoner → enabled
        #     - V4 原生 → 不强制（服务端默认 enabled）
        #     - claude-* → enabled（OpenAI 兼容端点收到 claude 名字时同样开 reasoning）
        if "thinking" not in body:
            implicit = default_thinking_type(raw_model)
            if implicit is not None:
                body["thinking"] = {"type": implicit}

        # 0b. 模型名称规范化（reasoner/chat 都会被映射到 v4-flash）
        body["model"] = normalize_model_name(raw_model, self._model_routes_dicts)
        model = body.get("model", "")

        # 0c. Flash→Pro 选择性升格路由（仅 v4-flash + 启用时）
        #     在全部后续处理之前改写 model，让 thinking/sampling/skills 走 Pro 路径。
        if (
            self.config.flash_upgrade.enabled
            and model == V4_FLASH
        ):
            self._maybe_upgrade(body)
            model = body.get("model", "")

        # 1. 默认 reasoning_effort=max + thinking.type=enabled 注入
        #    （仅当未显式 disabled 且未指定）。
        #    官方文档：reasoning_effort 是 thinking 对象的子字段，不是顶层参数。
        #    同步显式注入 type=enabled，让步骤 8 的 ensure_reasoning_content_persistence
        #    退化为纯"补齐"逻辑，消除跨步隐式依赖。
        if is_v4_model(model):
            explicitly_disabled = is_thinking_disabled(body.get("thinking"))
            if not explicitly_disabled:
                td = ensure_thinking_dict(body)
                td.setdefault("type", "enabled")
                td.setdefault("reasoning_effort", "max")

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
            # 无 profile 时的安全回退：0.6 介于 precise (0.25-0.45) 与 creative (0.90-1.20) 之间
            body.setdefault("temperature", 0.6)
            body.setdefault("top_p", 0.95)

        # 3. 参数过滤 — 移除 DeepSeek 不支持的参数（仅 functions / user）
        if self.config.deepseek.strip_unsupported_params:
            body = strip_unsupported_params(body)

        # 4. 清理空 stream_options
        body = sanitize_stream_options(body)

        # 按 sampling_profile 推导优化模式（在 optimization 块外定义，避免步骤 7.5 的作用域脆弱性）
        _opt_mode = "creative" if isinstance(sampling_profile, CreativeSamplingConfig) else "coding"

        # 5. 廉价提示词优化 + 内置 skills（in-process，0 额外上游调用）
        if self.config.optimization.enabled:
            opt = self.config.optimization
            await apply_cheap_optimizations(
                body,
                mode=_opt_mode,
                # A. 通用风格
                avoid_negative_style=opt.avoid_negative_style,
                natural_temperament=opt.natural_temperament,
                contextual_register=opt.contextual_register,
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

        # 6. 动态短段注入（场景化 PUA-substance 提示词）
        #    必须在 apply_cheap_optimizations（含 LLM 压缩）之后执行，避免随机
        #    句子进入压缩缓存键、每请求刷新缓存。
        if (
            self.config.optimization.enabled
            and self.config.optimization.dynamic_baskets
            and not has_tools(body)
        ):
            scenario = _scenario_from_profile(sampling_profile)
            if scenario:
                paragraphs = _assemble_basket_paragraphs(
                    scenario,
                    writing_kind=self.config.optimization.writing_basket_kind,
                )
                if paragraphs:
                    messages = body.get("messages")
                    if isinstance(messages, list) and messages:
                        for para in paragraphs:
                            append_to_system_message(messages, para)

        # 7. 无厘头 expert priming（最后一步，system 最前插入）
        #    Always 全场景生效；不进压缩缓存键；每次随机 2 条
        if (
            self.config.optimization.enabled
            and self.config.optimization.silly_expert_priming
            and not has_tools(body)
        ):
            primings = _pick_silly_n(2)
            if primings:
                messages = body.get("messages")
                if isinstance(messages, list) and messages:
                    # 包装为带署名的"摘录式"段落组后整体 prepend
                    block = _wrap_silly_for_injection(primings)
                    if block:
                        prepend_to_system_message(messages, block)

        # 7.5 V4 <think> 角色沉浸引导（creative mode + 非 tools 场景）
        #     引导 <think> 推理层进入角色第一人称内心独白模式，
        #     使角色的情感推理真实化，输出自然带体温。
        #     注入位置：最后一条 user 消息末尾（与 V4 训练时的注入位置一致）。
        #     idempotent：已有 marker 则跳过。
        if (
            self.config.optimization.enabled
            and _opt_mode == "creative"
            and not has_tools(body)
        ):
            messages = body.get("messages")
            if isinstance(messages, list) and messages:
                injected = inject_inner_os_marker(messages)
                if injected:
                    logger.debug("已注入 V4 角色沉浸 marker")

        # 8. V4 多轮 reasoning 自愈：在全部消息修改之后执行，确保
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

    def _stash_pending_upgrade(
        self, body: dict[str, Any], messages: list[dict[str, Any]], turns: int
    ) -> None:
        """快照 fingerprint + last_user_hash，等上游成功后再 commit。

        必要性：set_remaining 直接写入会让失败的上游请求也"白扣"一个
        Pro 轮次额度。延迟到 chat_completions / 流式自然结束后提交，
        失败请求不污染 tracker。

        快照在此处取，因为 messages 会被 skills 阶段（RE2/readurls）改写，
        提交时直接读取已不可靠。
        """
        fp, last_user_h = UpgradeTracker.snapshot_keys(messages)
        body["_deepproxy_pending_upgrade"] = {
            "fingerprint": fp,
            "last_user_hash": last_user_h,
            "turns": turns,
        }

    def _commit_pending_upgrade(self, body: dict[str, Any]) -> None:
        """上游成功后提交挂起的升格记账（无挂起则空操作）。"""
        pending = body.get("_deepproxy_pending_upgrade")
        if not isinstance(pending, dict):
            return
        self._upgrade_tracker.set_remaining_by_key(
            pending["fingerprint"], pending["last_user_hash"], pending["turns"],
        )

    def _maybe_upgrade(
        self,
        body: dict[str, Any],
    ) -> None:
        """Flash→Pro 升格路由主逻辑（Layer 0–3，全部 upfront）。

        决策顺序（短路）：
          1. Sentinel 强制升格（最高优先级）
          2. 对话已升格（Layer 3 持久化）
          3. 启发式快速路径（Layer 1）
          4. Router 决策（Layer 0）

        持久化记账（set_remaining）通过 _stash_pending_upgrade 延迟到上游
        成功后由调用方 commit，避免失败请求白扣 Pro 槽位。
        """
        cfg = self.config.flash_upgrade
        messages = body.get("messages", [])

        # ── Step 1: Sentinel / extra_body 强制升格 ──
        if has_upgrade_sentinel(messages) or extra_body_requests_upgrade(body):
            logger.info("Sentinel 强制升格 → %s", V4_PRO)
            body["model"] = V4_PRO
            self._stash_pending_upgrade(body, messages, cfg.persist_turns)
            return

        # ── Step 2: 对话已处于升格状态（Layer 3 持久化） ──
        # 预检 throttle 冷却：throttle 触发后的 cooldown 期内必须强制 Flash，
        # 不能让 persist cache 越过 throttle。
        if self._upgrade_throttle.in_cooldown(messages):
            self._upgrade_throttle.should_throttle(messages, False)  # 推进冷却计数
            self._upgrade_tracker.clear(messages)
            logger.info("升格限流冷却中 → 强制 %s", V4_FLASH)
            return

        if self._upgrade_tracker.is_upgraded(messages):
            remaining = self._upgrade_tracker.remaining(messages)
            logger.info("持久升格命中 → %s（剩余 %d 轮）", V4_PRO, remaining)
            body["model"] = V4_PRO
            return

        # ── Step 3: 启发式快速路径（Layer 1） ──
        heuristic_result = compute_complexity_score(messages)
        did_upgrade = False
        if heuristic_result.score >= cfg.heuristic_threshold:
            did_upgrade = True
            logger.info("启发式升格: score=%s >= threshold=%s",
                        heuristic_result.score, cfg.heuristic_threshold)

        # ── Step 4: Router 决策（Layer 0） ──
        # 注：v4-flash 处理简单编码任务效果已极好，不再对 coding_port 做阈值优惠。
        if not did_upgrade:
            router_score = self._upgrade_router.score(messages, body=body)
            if router_score >= cfg.router_threshold:
                logger.info(
                    "Router 升格: score=%.3f >= threshold=%.2f "
                    "(heuristic=%.1f/10, user_msgs=%d, user_chars=%d)",
                    router_score, cfg.router_threshold,
                    heuristic_result.score, heuristic_result.user_msg_count, len(heuristic_result.user_text),
                )
                did_upgrade = True
            else:
                logger.info(
                    "保留 Flash: score=%.3f < threshold=%.2f (heuristic=%.1f/10) → %s",
                    router_score, cfg.router_threshold, heuristic_result.score, V4_FLASH,
                )

        # ── Step 5: 防重复刷屏（Layer 2） ──
        # Coding Agent 场景下同一复杂消息可能重复多次；连续 N 次相同
        # user 消息触发升格后，强制回退到 Flash 并冷却，避免浪费。
        if did_upgrade:
            if self._upgrade_throttle.should_throttle(messages, True):
                did_upgrade = False
                # 同步清掉持久升格 entry，否则下一轮 Step 2 会越过 throttle
                # 直接走 Pro，使 cooldown 失效。
                self._upgrade_tracker.clear(messages)
                logger.info(
                    "升格限流: 连续 %d 次触发 → 强制 Flash（冷却 %d 轮）",
                    self._upgrade_throttle._max, self._upgrade_throttle._cooldown,
                )
        else:
            self._upgrade_throttle.should_throttle(messages, False)

        if did_upgrade:
            body["model"] = V4_PRO
            self._stash_pending_upgrade(body, messages, cfg.persist_turns)

    def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if self.config.deepseek.enable_reasoning:
            response = process_reasoning_response(response)
        return response

    # ------------------------------------------------------------------
    # 端点方法（轻量封装，供 main.py 调用）
    # ------------------------------------------------------------------

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        request_messages = list(body.get("messages") or [])
        # 是否需要剥离 CoT Reflection 标签（由 apply_cheap_optimizations 在 prepare_request 时打的标）
        strip_cot = bool(body.get("_deepproxy_strip_cot", False))
        raw = await call_litellm(self.config, body)
        result = self.process_response(raw)
        if strip_cot:
            for choice in result.get("choices", []):
                msg = choice.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    msg["content"] = extract_cot_output(msg["content"])
        # 按对话前缀写缓存，供下一轮补齐
        self._reasoning_cache.remember_response(request_messages, result)
        # 上游成功，提交挂起的升格记账（失败路径会 raise，下方不会执行）
        self._commit_pending_upgrade(body)
        return result

    async def iter_chat_chunks(
        self, body: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """业务层流式 chunk 流（dict 形态）。

        - 每个 yield：OpenAI 风格的 chunk dict（带 reasoning 字段已自愈/累加）
          或 `{"error": {...}}` 错误终止
        - 自然结束 = 流正常完成
        - 期间累加 content / reasoning_content / tool_calls，结束后写 ReasoningCache

        SSE 序列化（`data:` 前缀、`[DONE]` 前哨）由调用方在协议层完成。
        """
        request_messages = list(body.get("messages") or [])
        accumulator = StreamingReasoningAccumulator(request_messages=request_messages)
        completed_cleanly = False
        saw_error_frame = False
        try:
            async for chunk_dict in iter_litellm_chunks(self.config, body, _accumulator=accumulator):
                if isinstance(chunk_dict.get("error"), dict) and not chunk_dict.get("choices"):
                    saw_error_frame = True
                yield chunk_dict
            completed_cleanly = True
        finally:
            accumulator.flush_to_cache(self._reasoning_cache)
            # 流自然结束（无 error frame、无异常、未被取消）才提交升格记账
            if completed_cleanly and not saw_error_frame:
                self._commit_pending_upgrade(body)

    async def chat_completions_stream(
        self, body: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """OpenAI 协议层流式输出：iter_chat_chunks → SSE 字符串。

        负责协议细节：dict → `data: {...}\\n\\n`、错误帧序列化、`data: [DONE]\\n\\n` 前哨。
        """
        async for item in self.iter_chat_chunks(body):
            yield f"data: {json.dumps(item)}\n\n"
            if isinstance(item.get("error"), dict) and not item.get("choices"):
                yield SSE_DONE
                return
        yield SSE_DONE

    async def list_models(self) -> dict[str, Any]:
        """列出可用模型（同时兼容 OpenAI / OpenRouter / Anthropic 三种生态）。

        优先从 DeepSeek 上游 `GET /v1/models` 拉取真实清单；上游不可用时退化到
        内置 V4 模型列表（含 `[1m]` 变体）。`expose_legacy_models=True` 会附加老别名；
        `model_routes` 中的自定义对外名也会合并进去（去重）。

        响应同时含 OpenAI 的 `object=list` 和 Anthropic 的 `first_id/last_id/has_more`
        分页字段；条目层 normalize_model_entry 同时输出两套生态字段。
        """
        raw = await fetch_upstream_models(
            self.config.deepseek.api_key,
            self.config.deepseek.api_base,
            self._get_http_client(),
        )
        models = build_models_list(
            raw,
            expose_legacy_models=self.config.deepseek.expose_legacy_models,
            model_routes=self._model_routes_dicts,
        )
        return {
            # OpenAI 列表标识
            "object": "list",
            "data": models,
            # Anthropic 分页字段（无后续页 → first/last 为首尾 id，has_more=false）
            "first_id": models[0]["id"] if models else None,
            "last_id": models[-1]["id"] if models else None,
            "has_more": False,
        }
