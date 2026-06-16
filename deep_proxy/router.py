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

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncGenerator, Callable
from uuid import uuid4

import httpx
from fastapi import HTTPException

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
from .utils import (
    SSE_DONE, append_to_system_message, is_error_frame, is_heartbeat,
    prepend_to_system_message,
)
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
from .cross_consult.streaming import aggregate_stream_to_response, stream_aggregated_call
from .cross_consult.client_stream import (
    TurnResult,
    make_terminal_frame,
    stream_cross_consult_continuation,
    stream_one_turn,
    stream_turn_with_retry,
    stream_with_retry,
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


def _message_to_chunk_frames(
    message: dict[str, Any],
    *,
    frame_id: str | None = None,
    model: str | None = None,
    created: int | None = None,
) -> list[dict[str, Any]]:
    """把一条（StyleGuard 修正后的）assistant 消息转换为流式 chunk 帧序列。

    用于流式路径在 post-stream 修正后，以修正消息为唯一真相重建待 yield 的缓冲帧——
    避免保留与修正内容不一致的原始帧（如修正后 tool_calls 改变、原帧已过期）。

    OpenAI 一致性：同一 completion 的所有 chunk **共享一个 id**（不再每帧新 uuid）；
    `role:"assistant"` 仅落在首帧 delta；尽量沿用上游的 model/created。
    finish_reason **由修正消息自身**派生（有 tool_calls → "tool_calls"，否则 "stop"），
    不沿用上游原始 finish_reason——否则参数违规修正把 tool_call 改写成纯文本后，
    仍带 finish_reason="tool_calls" 会误导客户端 SDK 等待并不存在的工具调用。
    """
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls")
    finish_reason = "tool_calls" if tool_calls else "stop"
    fid = frame_id or f"chatcmpl-{uuid4().hex[:29]}"

    def _frame(delta: dict[str, Any]) -> dict[str, Any]:
        f: dict[str, Any] = {"id": fid, "object": "chat.completion.chunk"}
        if model is not None:
            f["model"] = model
        if created is not None:
            f["created"] = created
        f["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
        return f

    frames: list[dict[str, Any]] = []
    if content or reasoning:
        delta: dict[str, Any] = {"content": content}
        if reasoning:
            delta["reasoning_content"] = reasoning
        frames.append(_frame(delta))
    if tool_calls:
        frames.append(_frame({"tool_calls": tool_calls}))
    if not frames:
        frames.append(_frame({"content": ""}))  # 兜底：空内容也发一帧承载 finish_reason

    # role 仅落首帧 delta；finish_reason 落末帧
    frames[0]["choices"][0]["delta"]["role"] = "assistant"
    frames[-1]["choices"][0]["finish_reason"] = finish_reason
    return frames


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

        # 0e. 全模式强制 thinking=enabled（force_thinking_enabled，默认开）：覆盖客户端显式
        #     disabled 与 deepseek-chat 别名的 disabled。须在 step 1（reasoning_effort 注入，
        #     按 is_thinking_disabled 门控）之前，让被覆盖的请求也注入 reasoning_effort。
        #     仅对支持 thinking 的 provider（has_thinking_param / 老路径 V4）生效。
        #     根因：DeepSeek 无法真正 disabled（LiteLLM 丢弃 {type:disabled} → 服务端默认
        #     enabled），强制 enabled 让代理状态与上游一致。
        if self.config.force_thinking_enabled:
            supports_thinking = (
                provider.has_thinking_param if provider is not None else is_v4_model(model)
            )
            if supports_thinking:
                ensure_thinking_dict(body)["type"] = "enabled"

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

        # 8. Cross-Consult 工具注入（在所有 skills 之后，避免改变 has_tools 影响其它步骤）。
        #    **必须在 reasoning 自愈之前**：自愈的 ReasoningCache 键含 system prefix，而
        #    flush（iter_chat_chunks 在 prepare_request 全部完成后捕获 request_messages）看到
        #    的是 cc 注入**后**的 system；若自愈/backfill 在 cc 注入前算键，则 flush 与 backfill
        #    的 system 不一致（差 cc 增量）→ 缓存对历史 assistant 永远 miss → 退化 dummy。
        if (
            self.config.cross_consult.enabled
            and provider is not None
        ):
            inject_into_request(
                body,
                source_provider_name=provider.name,
                cc_config=self.config.cross_consult,
            )

        # 9. V4 多轮 reasoning 自愈：在全部消息修改（含 cc 注入）之后执行，确保缓存键 prefix
        #    与 remember_response / 流式 flush 存储时的对话前缀（同样含 cc 注入）一致。
        #    provider 给定时走 provider.has_reasoning_content，否则保持老 V4 判定。
        has_rc = provider.has_reasoning_content if provider is not None else is_v4_model(model)
        if has_rc:
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

    async def _apply_style_guard(
        self,
        body: dict[str, Any],
        provider: Provider | None,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """对 assistant 响应运行 StyleGuard，违规时反馈重发。

        共享于流式 (iter_chat_chunks) 与非流式 (chat_completions) 两条路径。
        返回处理后的 result（无违规时原值返回；违规修正后为新响应）。
        """
        from .optimization.style_guard import (
            apply_style_guard_loop, apply_fluency_fix, RULES,
            _has_override_tag, _strip_override_tag,
        )
        # 显式跳过标签：清理后跳过 StyleGuard 与 fluency（与流式 _styleguard_scan_stream 一致）。
        _ovc = result.get("choices") or []
        if _ovc:
            _ovm = _ovc[0].get("message") or {}
            _ovtext = _ovm.get("content") or ""
            if _has_override_tag(_ovtext):
                _ovm["content"] = _strip_override_tag(_ovtext)
                return self.process_response(result, provider=provider)
        async def _resend():
            return await call_litellm(self.config, body, provider=provider)
        # 异族模型调用：跨家族修正，解决同模型对自身违规不敏感的问题
        call_alt: Callable | None = None
        if (
            self.config.cross_consult.enabled
            and provider is not None
            and self.config.cross_consult.pair_for(provider.name)
        ):
            _alt_name = self.config.cross_consult.pair_for(provider.name)
            _alt_provider = self.config.providers.get(_alt_name)
            if _alt_provider and _alt_provider.api_key:
                async def _resend_alt():
                    # 异族调用：暂存原始模型名，替换为异族 provider 的模型
                    _saved_model = body.get("model")
                    body["model"] = _alt_provider.flash_model
                    # 异族 provider 的 thinking 参数格式可能不同，由 call_litellm 的
                    # _assemble_litellm_body 处理 allowed_extra_params 透传
                    try:
                        return await call_litellm(self.config, body, provider=_alt_provider)
                    finally:
                        body["model"] = _saved_model
                call_alt = _resend_alt
        corrected = await apply_style_guard_loop(
            body=body,
            call_upstream=_resend,
            result=result,
            rules=RULES,
            max_retries=self.config.style_guard.max_retries,
            call_alt_upstream=call_alt,
        )
        # AI 通顺性审查：仅对含叙事锚点词的文本触发
        corrected = await apply_fluency_fix(
            body=body,
            call_upstream=_resend,
            result=corrected,
        )
        return self.process_response(corrected, provider=provider)

    @staticmethod
    def _slot_message(
        content: str, reasoning: str, tool_calls: Any = None,
    ) -> dict[str, Any]:
        """从累加槽构造一条 assistant 消息（供 StyleGuard 合成 result）。

        tool_calls 必须随之传入——否则 apply_style_guard_loop 看不到 tool_calls，
        会把纯 tool_call（无文本）响应当作空响应直接跳过，漏扫 Edit/Write 参数违规。
        """
        msg: dict[str, Any] = {"content": content, "role": "assistant"}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    @staticmethod
    def _rebuild_stream_frames(
        message: dict[str, Any], buffered: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """以修正消息重建缓冲帧，并携带原始流的 usage（若客户端请求 include_usage）。

        非流式路径会保留 usage；流式修正若丢弃 usage 会让 include_usage 客户端收不到
        用量。只提取原始 usage **对象**、重新封装为干净的 usage-only 帧（choices=[]、
        无 finish_reason），而非转发原始 usage chunk——后者带自身 finish_reason 与
        已被修正取代的 stale choices，会造成"两个终止帧 + 泄漏违规原文"。
        """
        # 沿用上游真实 id/model/created——同一 completion 全部 chunk 必须共享 id，
        # 且重建帧补回 model/created 以满足严格客户端 SDK 的 schema。
        frame_id = next((f.get("id") for f in buffered if f.get("id")), None) \
            or f"chatcmpl-{uuid4().hex[:29]}"
        model = next((f.get("model") for f in buffered if f.get("model")), None)
        created = next((f.get("created") for f in buffered if f.get("created")), None)
        frames = _message_to_chunk_frames(
            message, frame_id=frame_id, model=model, created=created,
        )
        usage = next((f["usage"] for f in buffered if f.get("usage")), None)
        if usage is not None:
            usage_frame: dict[str, Any] = {
                "id": frame_id, "object": "chat.completion.chunk",
            }
            if model is not None:
                usage_frame["model"] = model
            if created is not None:
                usage_frame["created"] = created
            usage_frame["choices"] = []
            usage_frame["usage"] = usage
            frames.append(usage_frame)
        return frames

    async def _styleguard_scan_stream(
        self,
        body: dict[str, Any],
        provider: Provider | None,
        accumulator: StreamingReasoningAccumulator,
        buffered: list[dict[str, Any]],
        finish_reason: str | None,
    ) -> list[dict[str, Any]]:
        """流式 post-stream StyleGuard：扫描累加结果，必要时修正并重建缓冲帧。

        扫描 content + tool_call 参数（与非流式一致，含纯 tool_call 无文本场景）。
        修正后以 `_message_to_chunk_frames` 从修正消息重建帧，避免保留与修正不一致的
        原始帧。异常时回滚注入的反馈消息并回退到原始缓冲帧。
        """
        from .optimization.style_guard import (
            RULES, apply_fluency_fix,
            scan_tool_call_violations, scan_violations,
            _has_override_tag, _strip_override_tag,
        )

        # 全程包在 try 内——扫描/正则/槽提取的任何异常都只降级回退到原始缓冲帧，
        # 绝不让 post-stream 处理异常撕裂已完整生成的整条响应流。
        msgs_before = len(body.get("messages", []))
        try:
            slot = accumulator.get_slot(0)
            content = slot.get("content", "") or ""
            reasoning = slot.get("reasoning_content", "") or ""
            tool_calls = slot.get("tool_calls")
            if not content and not tool_calls:
                return buffered

            # 显式跳过标签：清理后直通，跳过 StyleGuard 与 fluency（与非流式 _apply_style_guard
            # 一致）。以修正消息重建帧——稳健处理标签被拆到多个 delta 帧的情形。
            if content and _has_override_tag(content):
                clean = _strip_override_tag(content)
                accumulator.update_slot_content(0, clean, reasoning, tool_calls=tool_calls)
                return self._rebuild_stream_frames(
                    self._slot_message(clean, reasoning, tool_calls), buffered,
                )

            violations = scan_violations(content, RULES)
            tc_violations = scan_tool_call_violations(tool_calls, RULES)
            # 合成一条与上游响应等价的 result，供 StyleGuard / fluency 复用非流式逻辑
            result = {"choices": [{
                "index": 0,
                "message": self._slot_message(content, reasoning, tool_calls),
                "finish_reason": finish_reason or "stop",
            }]}

            if violations or tc_violations:
                corrected = await self._apply_style_guard(body, provider, result)
                choices = corrected.get("choices", [])
                if not choices or "message" not in choices[0]:
                    raise ValueError("StyleGuard 返回空 choices")
                new_msg = choices[0]["message"]
                new_content = new_msg.get("content", "") or ""
                new_tc = new_msg.get("tool_calls")
                # 内容或 tool_calls 发生变化 → 以修正消息重建缓冲帧
                if new_content != content or new_tc != tool_calls:
                    accumulator.update_slot_content(
                        0, new_content, new_msg.get("reasoning_content", "") or "",
                        tool_calls=new_tc,
                    )
                    return self._rebuild_stream_frames(new_msg, buffered)
                return buffered

            # 无违规：AI 通顺性审查（与非流式 _apply_style_guard 内的 fluency 一致）。
            # apply_fluency_fix 内部就地审查工具写入正文 + 随附 prose，并保留 tool_calls；
            # 故此处对含/不含 tool_calls 的响应统一处理，从修正后整条消息重建帧。
            # tool_calls 参数是**就地**改写的，须在审查前快照序列化，否则改后比较同一对象
            # 永远相等 → 漏判变化、返回过期原帧。
            _pre_tc = json.dumps(tool_calls, ensure_ascii=False, sort_keys=True) \
                if tool_calls else None
            fluent = await apply_fluency_fix(
                body=body,
                call_upstream=lambda: call_litellm(self.config, body, provider=provider),
                result=result,
            )
            fluent = self.process_response(fluent, provider=provider)
            choices = fluent.get("choices", [])
            if choices and "message" in choices[0]:
                f_msg = choices[0]["message"]
                f_content = f_msg.get("content", "") or ""
                f_tc = f_msg.get("tool_calls")
                _post_tc = json.dumps(f_tc, ensure_ascii=False, sort_keys=True) \
                    if f_tc else None
                if f_content != content or _post_tc != _pre_tc:
                    accumulator.update_slot_content(
                        0, f_content, f_msg.get("reasoning_content", "") or "",
                        tool_calls=f_tc,
                    )
                    return self._rebuild_stream_frames(f_msg, buffered)
            return buffered
        except Exception:
            # 回滚可能注入的反馈消息（loop/fluency 多已自恢复，此处兜底异常中途退出）
            if "messages" in body and len(body["messages"]) > msgs_before:
                del body["messages"][msgs_before:]
            logger.warning(
                "StyleGuard 流式扫描异常，回滚反馈消息，回退到原始响应", exc_info=True,
            )
            return buffered

    # ------------------------------------------------------------------
    # 端点方法（轻量封装，供 main.py 调用）
    # ------------------------------------------------------------------

    async def chat_completions(
        self, body: dict[str, Any], *, provider: Provider | None = None,
    ) -> dict[str, Any]:
        request_messages = list(body.get("messages") or [])
        # 是否需要剥离 CoT Reflection 标签（由 apply_cheap_optimizations 在 prepare_request 时打的标）
        strip_cot = bool(body.get("_deepproxy_strip_cot", False))

        # Cross-Consult 活跃时：初始调用也走流式聚合（aggregate_stream_to_response），
        # 避免深度思考在非流式 litellm.acompletion() 中无超时保护导致客户端墙钟超时。
        # 注入 tools + awareness prompt 后，模型 reasoning 可能远超普通请求；
        # 流式路径的 idle_timeout / first_chunk_timeout 提供 chunk 级守护。
        cc_active = (
            self.config.cross_consult.enabled
            and provider is not None
            and self.config.cross_consult.pair_for(provider.name) is not None
        )

        sc = self.config.streaming
        cc_idle = float(sc.idle_timeout_seconds)
        cc_first = float(sc.first_chunk_timeout_seconds)
        cc_reasoning = float(sc.reasoning_idle_timeout_seconds)
        cc_hb = float(sc.heartbeat_seconds)
        if cc_active:
            raw = await aggregate_stream_to_response(
                self.config, body, provider=provider,
                idle_timeout=cc_idle, first_chunk_timeout=cc_first,
                reasoning_idle=cc_reasoning, heartbeat_seconds=cc_hb,
            )
            if "_dp_error" in raw:
                raise HTTPException(status_code=504, detail={
                    "error": {
                        "message": f"上游超时: {raw['_dp_error']}",
                        "type": "timeout_error",
                        "param": None, "code": 504,
                    }
                })
        else:
            raw = await call_litellm(self.config, body, provider=provider)

        result = self.process_response(raw, provider=provider)

        # Cross-Consult 拦截：若响应含 cross_consult tool_call，执行 consult + 重发循环。
        # 重发走流式聚合（stream_aggregated_call）以避免深度思考触发墙钟超时；返回形状
        # 保持非流式 dict，process_response 无需感知差异。
        if cc_active:
            async def _resend_via_stream(cfg, b, *, provider=None):
                return await stream_aggregated_call(
                    cfg, b, provider=provider,
                    idle_timeout=cc_idle, first_chunk_timeout=cc_first,
                    reasoning_idle=cc_reasoning, heartbeat_seconds=cc_hb,
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
        # ── StyleGuard：响应侧风格扫描 + 反馈重发循环 ──
        if self.config.style_guard.enabled:
            result = await self._apply_style_guard(body, provider, result)
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

        控制流：按 cc_active 选 cc / 普通子生成器，顶层只做统一的脏退出处理
        （error frame 透传并标脏 saw_error_frame）+ finally 写 ReasoningCache 并在干净
        完成时提交升格记账。两条分支的细节封装进各自子生成器。超时收尾由
        stream_turn_with_retry 直接发硬错误帧（is_error_frame），无需哨兵。
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

        sub = (
            self._iter_cc_chunks(body, provider, accumulator) if cc_active
            else self._iter_plain_chunks(body, provider, accumulator)
        )

        # StyleGuard：流式路径启用时缓冲所有非心跳帧，流结束后扫描再 yield。
        # 非流式路径的 StyleGuard 在 chat_completions() 中已有；流式路径此前缺失。
        # 心跳帧立即透传（保持 SSE 连接存活），避免缓冲期 idle-read timeout。
        _sg_active = self.config.style_guard.enabled
        _buffered: list[dict[str, Any]] = []
        _last_finish_reason: str | None = None

        try:
            async for frame in sub:
                if is_error_frame(frame):
                    saw_error_frame = True
                # 心跳帧立即透传（保活信号），其余缓冲用于 post-stream 扫描
                if _sg_active and not is_heartbeat(frame):
                    _buffered.append(frame)
                    # 捕获上游 finish_reason（非心跳帧的 choice 中）
                    for _fc in frame.get("choices", []):
                        _fr = _fc.get("finish_reason")
                        if _fr:
                            _last_finish_reason = _fr
                else:
                    yield frame

            # ── StyleGuard 流式后置扫描 ──
            # 扫描累加结果（content + tool_call 参数），必要时修正并以"修正消息"
            # 为唯一真相重建缓冲帧。与非流式 chat_completions 共用 _apply_style_guard /
            # apply_fluency_fix，保证两条路径行为一致。
            # 扫描期间会做多次阻塞上游重发——其间发 SSE 心跳，防客户端 idle-read 超时
            # （缓冲已延迟全部输出，retry 阶段若静默易触发断连）。
            if _sg_active and not saw_error_frame:
                _sc = self.config.streaming
                _hb = float(_sc.heartbeat_seconds)
                # 整体墙钟上限：扫描内部 call_litellm 重发本身无 idle 守护，封顶防上游
                # 挂死时无限发心跳。预算 ≈ 每个可能重发各给一个 first_chunk 超时窗口。
                _scan_budget = float(_sc.first_chunk_timeout_seconds) * (
                    self.config.style_guard.max_retries + 8
                )
                _scan = asyncio.ensure_future(self._styleguard_scan_stream(
                    body, provider, accumulator, _buffered, _last_finish_reason,
                ))
                _waited = 0.0
                try:
                    while True:
                        try:
                            # shield：超时只中断等待，不取消扫描任务（保留重发进度）
                            _buffered = await asyncio.wait_for(
                                asyncio.shield(_scan), timeout=_hb,
                            )
                            break
                        except asyncio.TimeoutError:
                            _waited += _hb
                            if _waited >= _scan_budget:
                                # 扫描超总预算（上游疑似挂死）：取消并回退到原始缓冲帧
                                logger.warning(
                                    "StyleGuard 流式扫描超总预算 %.0fs，回退到原始响应",
                                    _scan_budget,
                                )
                                _scan.cancel()
                                with contextlib.suppress(BaseException):
                                    await _scan  # drain：避免 "exception never retrieved"
                                break
                            yield {"_dp_heartbeat": True}
                except BaseException:
                    # 客户端断开 / 生成器被关闭：取消并 drain 游离的扫描任务，避免泄漏
                    _scan.cancel()
                    with contextlib.suppress(BaseException):
                        await _scan
                    raise

            # yield 缓冲帧（原始或修正后）
            if _sg_active:
                for frame in _buffered:
                    yield frame

            # 流 + post-stream 扫描 + 缓冲全部交付完成才算干净完成——
            # 扫描期间客户端断开（BaseException）会跳过此处，finally 不提交升格记账，
            # 与"原始流式途中断开不提交"保持一致。
            completed_cleanly = True

        finally:
            accumulator.flush_to_cache(self._reasoning_cache)
            if completed_cleanly and not saw_error_frame:
                self._commit_pending_upgrade(body)

    async def _iter_cc_chunks(
        self,
        body: dict[str, Any],
        provider: Provider | None,
        accumulator: StreamingReasoningAccumulator,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """cross_consult 真流式子生成器：初始轮逐 token 透传 + 抑制 cc 工具帧 + 心跳，
        经 stream_turn_with_retry（pre-content 重试 + 硬错误）收尾，据胜出轮结果进入
        continuation / 终轮。

        超时收尾由 stream_turn_with_retry 直接发硬错误帧（is_error_frame），真实上游
        error frame 逐帧透传——两者皆经 iter_chat_chunks 的 is_error_frame 标脏（不提交
        升格记账）。
        """
        cc = self.config.cross_consult
        sc = self.config.streaming
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult):
            accumulator.restore(snap)   # 丢弃失败尝试的累加，保留更早内容（初始轮 snap 为空）
            return stream_one_turn(
                iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                ),
                turn, tool_name=cc.tool_name,
                idle_timeout=float(sc.idle_timeout_seconds),
                reasoning_idle=float(sc.reasoning_idle_timeout_seconds),
                first_chunk_timeout=float(sc.first_chunk_timeout_seconds),
                heartbeat_seconds=float(sc.heartbeat_seconds),
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_retries=int(sc.max_retries),
            on_result=lambda t: captured.__setitem__("turn", t),
        ):
            yield frame
        turn = captured.get("turn")
        if turn is None or turn.errored:
            # 硬错误（已发 error frame）或真实上游 error frame（已逐帧透传）→ 终止。
            return
        if not turn.had_cc_call:
            # 无 cc 调用：终轮，补发 finish_reason / 非 cc tool_calls
            yield make_terminal_frame(turn.finish_reason, turn.accumulated_tool_calls)
            return
        # 进入 continuation（其自身在重发轮硬错误/真实 error 时直接收尾）
        async for frame in stream_cross_consult_continuation(
            initial_tool_calls=turn.accumulated_tool_calls,
            body=body, source_provider=provider, config=self.config,
            cc_config=cc, accumulator=accumulator, initial_content=turn.content,
        ):
            yield frame

    async def _iter_plain_chunks(
        self,
        body: dict[str, Any],
        provider: Provider | None,
        accumulator: StreamingReasoningAccumulator,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """普通（非 cross_consult）流式子生成器：经 stream_with_retry 原样透传上游 chunk
        （含 tool_calls / finish_reason），并施加 mid-stream 超时**重试 + 硬错误**策略——

          - pre-content stall（首 chunk 前 / 推理中）→ 重发原请求（对客户端无缝，仅见心跳），
            最多 max_retries 次。健康流（持续产出）永不被打断。
          - post-content stall / 重试耗尽 → 发**硬错误帧**（`{"error": {...}}`，经
            is_error_frame 透传给客户端使 SDK 抛错；iter_chat_chunks 据此标脏不提交升格记账）。

        旧"注入'请重试' content + clean stop"对 agent 结构上不可能触发重试（clean stop =
        成功轮），已废弃。见 docs/superpowers/specs/2026-06-04-mid-stream-timeout-retry-design.md。
        """
        sc = self.config.streaming

        def make_upstream() -> AsyncGenerator[dict[str, Any], None]:
            # 每次尝试重建全新上游流（pre-content 重发的前提）。先 reset 共享 accumulator，
            # 否则废弃尝试的 reasoning_content / content 会与重试尝试拼接后污染 ReasoningCache。
            accumulator.reset()
            return iter_litellm_chunks(
                self.config, body, _accumulator=accumulator, provider=provider,
            )

        async for chunk_dict in stream_with_retry(
            make_upstream,
            idle_timeout=float(sc.idle_timeout_seconds),
            reasoning_idle=float(sc.reasoning_idle_timeout_seconds),
            first_chunk_timeout=float(sc.first_chunk_timeout_seconds),
            heartbeat_seconds=float(sc.heartbeat_seconds),
            max_retries=int(sc.max_retries),
        ):
            yield chunk_dict

    async def chat_completions_stream(
        self, body: dict[str, Any], *, provider: Provider | None = None,
    ) -> AsyncGenerator[str, None]:
        """OpenAI 协议层流式输出：iter_chat_chunks → SSE 字符串。

        负责协议细节：dict → `data: {...}\\n\\n`、错误帧序列化、`data: [DONE]\\n\\n` 前哨。
        """
        async for item in self.iter_chat_chunks(body, provider=provider):
            if is_heartbeat(item):
                # SSE 注释帧：规范明确忽略 `:` 开头行，零风险污染 delta 解析
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if is_error_frame(item):
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
