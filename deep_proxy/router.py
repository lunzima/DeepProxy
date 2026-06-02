"""核心请求路由器。

统一请求/响应管道（DeepSeek V4 + MiMo 双 provider）：

  Chat 端点 → prepare_request（模型名 / thinking / 采样 / 参数过滤 / skills 优化 /
              flash_upgrade 升格 + Direction C hysteresis 主动降格 / cross_consult 注入）
            → LiteLLM (acompletion / acompletion stream)
            → process_response（reasoning_content 兼容字段）
            → [可选] cross_consult 响应拦截 + 重发循环（异家族 pro 模型咨询）

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
from .deepseek_models import V4_FLASH
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
from .providers import Provider
from .utils import SSE_DONE, append_to_system_message, merge_tool_call_deltas, prepend_to_system_message
from .litellm_client import call_litellm, iter_litellm_chunks, _to_litellm_api_base
from .models_list import build_models_list, fetch_upstream_models
from .optimization import apply_cheap_optimizations, extract_cot_output, sample_in_range
from .optimization.compressor import SystemPromptCompressor
from .optimization.strip_telemetry import strip_telemetry_from_messages
from .optimization.dynamic_baskets import (
    assemble_paragraphs as _assemble_basket_paragraphs,
    scenario_from_profile as _scenario_from_profile,
)
from .compatibility.mimo_fixes import inject_top_level_reasoning_effort
from .cross_consult import RedirectTracker
from .cross_consult.interceptor import (
    execute_cross_consult_loop,
    inject_into_request,
)
from .cross_consult.streaming import stream_aggregated_call
from .cross_consult.client_stream import (
    STREAM_ERRORED,
    TurnResult,
    make_terminal_frame,
    stream_cross_consult_continuation,
    stream_one_turn,
)
from .optimization.dynamic_threshold import DynamicThresholdController
from .optimization.flash_upgrade import (
    RepeatUpgradeThrottle,
    UpgradeTracker,
)
from .optimization.upgrade_decision import UpgradeDecisionEngine
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
        # 升格决策引擎封装 5 步策略（_maybe_upgrade 仅作 shim）
        self._upgrade_engine = UpgradeDecisionEngine(
            cfg=config.flash_upgrade,
            upgrade_tracker=self._upgrade_tracker,
            throttle=self._upgrade_throttle,
            bert_router=self._upgrade_router,
        )
        # cross_consult 标签触发的整轮 provider 重定向跟踪器
        self._redirect_tracker = RedirectTracker()
        # Per-port 动态阈值控制器注册表（惰性创建，仅 dynamic_threshold.enabled 时）
        self._threshold_controllers: dict[int, DynamicThresholdController] = {}
        # LLM-based system prompt 压缩器（持久化磁盘缓存）
        # 复用 PreciseSamplingConfig 的采样预设：高确定性 + 微抖动，最适合
        # 同义改写类任务（确定性是主要诉求，微随机仅供并行重试）
        #
        # 凭据派生：compressor_model 默认带 deepseek/ 前缀 → 用 deepseek 凭据。
        # 新格式（providers 块）优先；老格式（顶层 deepseek:）兜底——避免新
        # 格式用户因顶层 deepseek 字段空而被误判为"未配置"。
        # 仅当解析后的 api_key 可用时初始化压缩器，否则降级 skip + 显式 warn
        # 避免每请求静默 401 → 用户付全 prompt tokens 而无感知。
        self._compressor: SystemPromptCompressor | None = None
        if config.optimization.enabled and config.optimization.compress_skills:
            ds_provider = config.providers.get("deepseek")
            ds_api_key = (
                (ds_provider.api_key if ds_provider else "")
                or config.deepseek.api_key
            )
            ds_api_base = (
                (ds_provider.api_base if ds_provider else "")
                or config.deepseek.api_base
            )
            if not ds_api_key:
                logger.warning(
                    "compress_skills=True 但未配置 DeepSeek api_key——"
                    "system prompt 压缩器降级为禁用（避免每请求静默 401）。"
                    "若需启用压缩，请在 providers.deepseek.api_key 或顶层 "
                    "deepseek.api_key 提供凭据（当前 compressor 仅支持 DeepSeek 凭据；"
                    "MiMo provider 凭据复用尚未实现）。",
                )
            else:
                from pathlib import Path
                self._compressor = SystemPromptCompressor(
                    cache_path=Path(config.optimization.compressor_cache_path),
                    api_key=ds_api_key,
                    api_base=_to_litellm_api_base(ds_api_base),
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

    def health_snapshot(self) -> dict[str, int]:
        """对外暴露 router 内部计数器供 /health 端点使用，避免 main.py 反复
        穿透到内部私有属性。"""
        return {
            "reasoning_cache_size": len(self._reasoning_cache),
            "upgrade_tracker_active": self._upgrade_tracker.active_count,
            "upgrade_throttle_size": self._upgrade_throttle.size,
            "redirect_tracker_active": self._redirect_tracker.active_count,
            "compressor_cache_entries": (
                self._compressor.cache_size if self._compressor is not None else 0
            ),
        }

    # ------------------------------------------------------------------
    # 预处理管道（公共方法，由 main.py 端点层调用）
    # ------------------------------------------------------------------

    async def prepare_request(
        self,
        body: dict[str, Any],
        *,
        sampling_profile: Any = None,
        provider: Provider | None = None,  # None 时走 deepseek 兼容行为
        port: int | None = None,           # 入站端口；解析 per-port 动态阈值控制器
    ) -> dict[str, Any]:
        """聊天补全请求预处理管道。

        重要：cross_consult 标签触发的整轮 provider 重定向（§12.11）发生在
        **本方法之前**，由 main.py::chat_completions 调 _maybe_redirect_provider
        完成。本方法收到的 provider 已是重定向后的目标，不再做二次检测。
        若未来新增直接调 prepare_request 的入口（绕过 chat_completions），
        必须显式先调 cross_consult.resolve_redirect。

        Args:
            sampling_profile: 若提供（PreciseSamplingConfig / CreativeSamplingConfig
                duck-typed），则强制覆盖 body 中的 4 个采样参数（不是 setdefault）。
                None 时退回旧的 creative_sampling.enabled-based 默认行为，便于测试。
            provider: Provider 实例或 None。给定时按 provider 配置路由 reasoning_effort
                注入位置；None 时退回老的 is_v4_model() 判定（向后兼容）。
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

        # 0b. 模型名称规范化
        #     - provider 已绑定且为非 deepseek：强制使用 provider 自家模型名
        #       （客户端可能传 deepseek-chat / claude-* / gpt-* 等任意名称；按 spec §7
        #        "port 决定一切"，这些都应路由到当前 provider 的 flash 档，由后续
        #        flash_upgrade 决定是否切 pro）。跳过 normalize_model_name，因为
        #        normalize 表是 DeepSeek 特定的。
        #     - provider 是 deepseek 或未绑定：保持老行为，走 normalize_model_name
        #       完成 legacy alias / clone alias 解析
        if provider is not None and provider.name != "deepseek":
            body["model"] = (
                raw_model
                if raw_model in (provider.flash_model, provider.pro_model)
                else provider.flash_model
            )
        else:
            body["model"] = normalize_model_name(raw_model, self._model_routes_dicts)
        model = body["model"]

        # 0c. 客户端 telemetry header 剥离（在升格哈希 / skills / 压缩缓存 key 之前）
        #     Claude Code 2.1.42+ 在 system 头部注入 `x-anthropic-billing-header: cc_version=...`
        #     含 session hash，每次新会话破坏 prefix cache。早期清理让所有下游看到稳定文本。
        #     生产路径下 main.py::chat_completions 已在 _maybe_redirect_provider 之前
        #     调过同一个 strip（确保 RedirectTracker fingerprint 稳定）；此处是
        #     直接调用 router.prepare_request 的测试场景的幂等 fallback。
        #     与 compressor 内部的 _normalize 形成三层防御。
        if (
            self.config.optimization.enabled
            and self.config.optimization.strip_client_telemetry
        ):
            messages = body.get("messages")
            if isinstance(messages, list):
                strip_telemetry_from_messages(messages)

        # 0d. Flash→Pro 选择性升格路由（仅 flash_model + 启用时）
        #     在全部后续处理之前改写 model，让 thinking/sampling/skills 走 Pro 路径。
        #     provider 给定时按 provider.flash_model 触发（支持 MiMo 等非 DeepSeek provider）；
        #     provider=None 时退回老路径（仅匹配硬编码 V4_FLASH）。
        upgrade_target = provider.flash_model if provider is not None else V4_FLASH
        if self.config.flash_upgrade.enabled and model == upgrade_target:
            self._maybe_upgrade(body, provider=provider, controller=self._controller_for_port(port))
            model = body.get("model", "")

        # 1. 默认 reasoning_effort 注入（仅当未显式 disabled 且未指定）
        #    DeepSeek: thinking.reasoning_effort = "max"（嵌套）
        #    MiMo:     reasoning_effort = "high"（顶层）
        #    取值与位置都从 provider 配置读取，不再硬编码
        if provider is not None and provider.has_thinking_param:
            explicitly_disabled = is_thinking_disabled(body.get("thinking"))
            if not explicitly_disabled:
                field_path = provider.reasoning_effort_field
                value = provider.reasoning_effort_value
                if field_path == "thinking.reasoning_effort":
                    td = ensure_thinking_dict(body)
                    td.setdefault("type", "enabled")
                    td.setdefault("reasoning_effort", value)
                elif field_path == "reasoning_effort":
                    inject_top_level_reasoning_effort(body, value=value)
                else:
                    logger.warning("未知 reasoning_effort_field: %s", field_path)
        elif is_v4_model(model):
            # provider=None（老路径）或 provider.has_thinking_param=False：仅对 V4 注入 thinking.reasoning_effort=max
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
            await apply_cheap_optimizations(
                body,
                opt=self.config.optimization,
                mode=_opt_mode,
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

        # 7.5 V4 <think> 角色沉浸引导（仅 DeepSeek + creative + 非 tools）
        #     spec §11: 该 skill 基于 DeepSeek V4 训练分布，对 MiMo 等异家族无意义
        #     引导 <think> 推理层进入角色第一人称内心独白模式，
        #     使角色的情感推理真实化，输出自然带体温。
        #     注入位置：最后一条 user 消息末尾（与 V4 训练时的注入位置一致）。
        #     idempotent：已有 marker 则跳过。
        is_deepseek_path = provider is None or provider.name == "deepseek"
        if (
            is_deepseek_path
            and self.config.optimization.enabled
            and _opt_mode == "creative"
            and self.config.optimization.inner_os_marker
            and not has_tools(body)
        ):
            messages = body.get("messages")
            if isinstance(messages, list) and messages:
                injected = inject_inner_os_marker(messages)
                if injected:
                    logger.debug("已注入 V4 角色沉浸 marker")

        # 8. V4 多轮 reasoning 自愈：在全部消息修改之后执行，确保
        #    缓存键与 remember_response 存储时的对话前缀一致。
        #    provider 给定时走 provider.has_reasoning_content，否则保持老 V4 判定。
        has_rc = provider.has_reasoning_content if provider is not None else is_v4_model(model)
        if has_rc:
            messages = body.get("messages", [])
            if messages:
                body = ensure_reasoning_content_persistence(
                    messages, body, cache=self._reasoning_cache,
                )

        # 9. Cross-Consult 工具注入（在所有 skills 之后，避免改变 has_tools 影响其它步骤）
        if (
            self.config.cross_consult.enabled
            and provider is not None
        ):
            inject_into_request(
                body,
                source_provider_name=provider.name,
                cc_config=self.config.cross_consult,
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

    def _controller_for_port(self, port: int | None) -> DynamicThresholdController | None:
        """按 port 惰性取/建动态阈值控制器；未启用或 port 为 None 返回 None。"""
        dt = self.config.flash_upgrade.dynamic_threshold
        if port is None or not dt.enabled:
            return None
        ctrl = self._threshold_controllers.get(port)
        if ctrl is None:
            ctrl = DynamicThresholdController(
                flash_floor=dt.flash_floor,
                band=dt.band,
                window=dt.window,
                kp=dt.kp,
                min_samples=dt.min_samples,
            )
            self._threshold_controllers[port] = ctrl
        return ctrl

    def _build_upgrade_router(self):
        """初始化升格决策器（Layer 0）。"""
        cfg = self.config.flash_upgrade
        if cfg.router_type == "bert" and cfg.bert_checkpoint:
            return create_router("bert", checkpoint_path=cfg.bert_checkpoint)
        return create_router("rule")

    def _commit_pending_upgrade(self, body: dict[str, Any]) -> None:
        """上游成功后提交挂起的升格记账（无挂起则空操作）。"""
        pending = body.get("_deepproxy_pending_upgrade")
        if not isinstance(pending, dict):
            return
        self._upgrade_tracker.set_remaining_by_key(
            pending["fingerprint"], pending["last_user_hash"], pending["turns"],
            provider=pending.get("provider", "deepseek"),
        )

    def _maybe_upgrade(
        self,
        body: dict[str, Any],
        *,
        provider: Provider | None = None,
        controller: DynamicThresholdController | None = None,
    ) -> None:
        """Flash→Pro 升格路由 — shim 委托给 UpgradeDecisionEngine。

        历史 API 保留（46 个 test_flash_upgrade 用例直接调用本方法，controller
        默认 None → 行为不变）；实际策略实现见 optimization/upgrade_decision.py。
        """
        self._upgrade_engine.apply(body, provider=provider, controller=controller)

    def process_response(
        self, response: dict[str, Any], *, provider: Provider | None = None,
    ) -> dict[str, Any]:
        has_rc = (
            provider.has_reasoning_content if provider is not None
            else self.config.deepseek.enable_reasoning
        )
        if has_rc:
            response = process_reasoning_response(response)
        return response

    # ------------------------------------------------------------------
    # 端点方法（轻量封装，供 main.py 调用）
    # ------------------------------------------------------------------

    async def chat_completions(
        self, body: dict[str, Any], *, provider: Provider | None = None,
    ) -> dict[str, Any]:
        request_messages = list(body.get("messages") or [])
        # 是否需要剥离 CoT Reflection 标签（由 apply_cheap_optimizations 在 prepare_request 时打的标）
        strip_cot = bool(body.get("_deepproxy_strip_cot", False))
        raw = await call_litellm(self.config, body, provider=provider)
        result = self.process_response(raw, provider=provider)
        # Cross-Consult 拦截：若响应含 cross_consult tool_call，执行 consult + 重发循环。
        # 重发走流式聚合（stream_aggregated_call）以避免深度思考触发墙钟超时；返回形状
        # 保持非流式 dict，process_response 无需感知差异。
        if self.config.cross_consult.enabled and provider is not None:
            cc_idle = float(self.config.cross_consult.call_timeout_seconds)
            cc_first = float(self.config.cross_consult.first_chunk_timeout_seconds)

            async def _resend_via_stream(cfg, b, *, provider=None):
                return await stream_aggregated_call(
                    cfg, b, provider=provider,
                    idle_timeout=cc_idle, first_chunk_timeout=cc_first,
                )

            result = await execute_cross_consult_loop(
                body=body,
                initial_response=result,
                source_provider=provider,
                config=self.config,
                cc_config=self.config.cross_consult,
                call_litellm_fn=_resend_via_stream,
                process_response_fn=self.process_response,
            )
            result = self.process_response(result, provider=provider)
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
        self, body: dict[str, Any], *, provider: Provider | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """业务层流式 chunk 流（dict 形态）。

        cross_consult enabled + pair 存在 + provider 给定时，
        content/reasoning_content chunk 即时透传到客户端；cross_consult tool_call
        帧被抑制，consult 执行期间发心跳帧，重发轮再次逐 token 透传。

        - 每个 yield：OpenAI 风格的 chunk dict（带 reasoning 字段已自愈/累加）、
          心跳哨兵 `{"_dp_heartbeat": True}`，或 `{"error": {...}}` 错误终止
        - 自然结束 = 流正常完成
        - 结束后写 ReasoningCache；升格记账在完全成功时提交

        SSE 序列化（`data:` 前缀、`[DONE]` 前哨）由调用方在协议层完成。
        """
        cc_active = (
            self.config.cross_consult.enabled
            and provider is not None
            and self.config.cross_consult.pair_for(provider.name) is not None
        )

        request_messages = list(body.get("messages") or [])
        accumulator = StreamingReasoningAccumulator(request_messages=request_messages)
        completed_cleanly = False
        saw_error_frame = False

        try:
            if cc_active:
                turn = TurnResult()
                initial_iter = iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                )
                idle = float(self.config.cross_consult.call_timeout_seconds)
                first = float(self.config.cross_consult.first_chunk_timeout_seconds)
                hb = float(self.config.cross_consult.stream_heartbeat_seconds)
                async for frame in stream_one_turn(
                    initial_iter, turn, tool_name=self.config.cross_consult.tool_name,
                    idle_timeout=idle, first_chunk_timeout=first, heartbeat_seconds=hb,
                ):
                    if isinstance(frame.get("error"), dict) and not frame.get("choices"):
                        saw_error_frame = True
                    yield frame
                if turn.errored:
                    # 初始轮超时/error（error frame 已透传或仅超时无帧）——非干净完成，
                    # 不提交升格记账。
                    saw_error_frame = True
                    return
                if not turn.had_cc_call:
                    # 无 cc 调用：终轮，补发 finish_reason / 非 cc tool_calls
                    yield make_terminal_frame(turn.finish_reason, turn.accumulated_tool_calls)
                else:
                    async for frame in stream_cross_consult_continuation(
                        initial_tool_calls=turn.accumulated_tool_calls,
                        body=body, source_provider=provider, config=self.config,
                        cc_config=self.config.cross_consult, accumulator=accumulator,
                        initial_content=turn.content,
                    ):
                        # 重发轮 errored 哨兵（单例，按 identity 识别）：标记不干净，
                        # 吞掉不透传客户端
                        if frame is STREAM_ERRORED:
                            saw_error_frame = True
                            continue
                        if isinstance(frame.get("error"), dict) and not frame.get("choices"):
                            saw_error_frame = True
                        yield frame
                completed_cleanly = True
            else:
                async for chunk_dict in iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                ):
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
        self, body: dict[str, Any], *, provider: Provider | None = None,
    ) -> AsyncGenerator[str, None]:
        """OpenAI 协议层流式输出：iter_chat_chunks → SSE 字符串。

        负责协议细节：dict → `data: {...}\\n\\n`、错误帧序列化、`data: [DONE]\\n\\n` 前哨。
        """
        async for item in self.iter_chat_chunks(body, provider=provider):
            if item.get("_dp_heartbeat"):
                # SSE 注释帧：规范明确忽略 `:` 开头行，零风险污染 delta 解析
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if isinstance(item.get("error"), dict) and not item.get("choices"):
                yield SSE_DONE
                return
        yield SSE_DONE

    async def _build_provider_models(self, provider: Provider | None) -> list[dict[str, Any]]:
        """构建单个 provider 的模型条目列表（不含响应外壳）。

        - provider.name == "mimo"：本地 MIMO_MODELS（跳过上游拉取）
        - provider=None 或 deepseek：上游拉取 + 本地兜底 + 别名/路由合并
        """
        if provider is not None and provider.name == "mimo":
            return build_models_list(raw=[], provider=provider)
        raw = await fetch_upstream_models(
            self.config.deepseek.api_key,
            self.config.deepseek.api_base,
            self._get_http_client(),
        )
        return build_models_list(
            raw,
            expose_legacy_models=self.config.deepseek.expose_legacy_models,
            model_routes=self._model_routes_dicts,
            provider=provider,
        )

    async def list_models(
        self,
        *,
        provider: Provider | None = None,
        pool_providers: list[Provider] | None = None,
    ) -> dict[str, Any]:
        """列出可用模型（同时兼容 OpenAI / OpenRouter / Anthropic 三种生态）。

        provider 给定时按 provider 派发列表：
          - provider.name == "mimo"：跳过上游拉取，直接用本地 MIMO_MODELS
          - provider=None 或 provider.name == "deepseek"：现有行为（上游拉取 + 本地兜底）

        pool_providers 给定时（writing-port 配置了 model_pool）：列出池内各 provider
        家族的**并集**，按 home provider（provider 参数）优先排序、按 id 去重。

        优先从 DeepSeek 上游 `GET /v1/models` 拉取真实清单；上游不可用时退化到
        内置 V4 模型列表（含 `[1m]` 变体）。`expose_legacy_models=True` 会附加老别名；
        `model_routes` 中的自定义对外名也会合并进去（去重）。

        响应同时含 OpenAI 的 `object=list` 和 Anthropic 的 `first_id/last_id/has_more`
        分页字段；条目层 normalize_model_entry 同时输出两套生态字段。
        """
        if pool_providers:
            # home 优先，其余按给定顺序追加（按 name 去重 provider）
            ordered_provs: list[Provider] = []
            seen_names: set[str] = set()
            for p in ([provider] if provider is not None else []) + list(pool_providers):
                if p is not None and p.name not in seen_names:
                    ordered_provs.append(p)
                    seen_names.add(p.name)
            models = []
            seen_ids: set[str] = set()
            for p in ordered_provs:
                for m in await self._build_provider_models(p):
                    if m["id"] not in seen_ids:
                        models.append(m)
                        seen_ids.add(m["id"])
        else:
            models = await self._build_provider_models(provider)
        return {
            # OpenAI 列表标识
            "object": "list",
            "data": models,
            # Anthropic 分页字段（无后续页 → first/last 为首尾 id，has_more=false）
            "first_id": models[0]["id"] if models else None,
            "last_id": models[-1]["id"] if models else None,
            "has_more": False,
        }
