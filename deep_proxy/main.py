"""DeepProxy FastAPI 应用入口。

暴露兼容 OpenAI API + Anthropic Messages API 格式的端点，按入站端口路由到上游 provider（DeepSeek / MiMo）。

统一请求管道：
  /v1/chat/completions → router.prepare_request（含廉价提示词优化 + Flash→Pro 升格）
                       → LiteLLM
                       → 后处理
  /v1/messages → claude_request_to_openai → router.prepare_request → LiteLLM
              → openai_response_to_claude / openai_stream_to_claude
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ProxyConfig
from .router import DeepProxyRouter

logger = logging.getLogger("deep_proxy")

config: ProxyConfig | None = None
router: DeepProxyRouter | None = None
_lifespan_done: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    global config, router, _lifespan_done

    # 双端口共享同一个 app 实例，两个 uvicorn Server 各触发一次 lifespan。
    # 首个到达的实例（_lifespan_done=False）独占完整的 startup + shutdown 流程
    # （加载配置、初始化 BERT 路由器、创建 HTTP 客户端等重量资源仅执行一次）；
    # 第二个实例直接 yield 返回，既不做初始化也不做清理，共享首个实例的全局状态。
    if _lifespan_done:
        yield
        return
    _lifespan_done = True

    import os

    loaded_config = ProxyConfig.discover_and_load()

    # 环境变量兜底
    if not loaded_config.deepseek.api_key:
        loaded_config.deepseek.api_key = os.getenv("DEEPSEEK_API_KEY", "")

    config = loaded_config
    router = DeepProxyRouter(config)

    if config.optimization.enabled:
        logger.info(
            "提示词优化已启用 (cot_reflection=%s, re2=%s, compress_skills=%s)",
            config.optimization.cot_reflection,
            config.optimization.re2,
            config.optimization.compress_skills,
        )
        if config.optimization.compress_skills:
            from pathlib import Path as _P
            cache_abs = _P(config.optimization.compressor_cache_path).resolve()
            logger.info("压缩缓存文件路径: %s", cache_abs)

    # 检查每个配置的 provider 是否有 api_key（多 provider 路径）
    for name, prov in config.providers.items():
        if not prov.api_key:
            logger.warning("provider %s 未设置 api_key！请通过 config.yaml 或环境变量配置。", name)

    logger.info(
        "DeepProxy 启动完成 — 监听 %s:%s (coding/precise) + %s:%s (writing/creative, basket=%s)",
        config.host, config.coding_port,
        config.host, config.writing_port,
        config.optimization.writing_basket_kind,
    )
    # 启动诊断横幅：明确打印加载到的配置里每个 port 的 model_pool 状态，
    # 便于确认运行进程到底加载了哪份配置（排查 round-robin 未生效）。
    for _b in config.ports:
        _pool = [(e.provider, e.model, e.weight) for e in (_b.model_pool or [])]
        logger.info(
            "[startup] port=%s provider=%s sampling=%s model_pool=%s",
            _b.port, _b.provider, _b.sampling,
            _pool if _pool else "（无）",
        )
    _dt = config.flash_upgrade.dynamic_threshold
    logger.info(
        "[startup] dynamic_threshold enabled=%s flash_floor=%s band=%s",
        _dt.enabled, _dt.flash_floor, _dt.band,
    )

    yield

    if router:
        await router.close()
    logger.info("DeepProxy 已关闭")


app = FastAPI(
    title="DeepProxy",
    description="多 provider FastAPI 代理（DeepSeek + MiMo + cross_consult）",
    version="0.1.0",
    lifespan=lifespan,
)

# 浏览器 / Electron 渲染进程客户端的 fetch 需要 CORS 头。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mask(token: str) -> str:
    """掩码 token 用于诊断日志：保留前 6 + 后 2 字符，中间打码。"""
    if not token:
        return "<empty>"
    if len(token) <= 10:
        return f"{token[:2]}***"
    return f"{token[:6]}...{token[-2:]} (len={len(token)})"


def _request_authorized(request: Request) -> bool:
    """同时识别两种认证头：OpenAI 风格 `Authorization: Bearer` 与 Anthropic 风格 `x-api-key`。

    /v1/models 同时面向 OpenAI 与 Anthropic 生态客户端（条目里两套字段共存），
    Claude Code 用 ANTHROPIC_API_KEY 配置时只发 x-api-key，因此 OpenAI 风格端点
    也必须接受这个头，否则启动期 /v1/models 探测就会 401。
    """
    if not (config and config.api_key):
        return True
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key == config.api_key:
        return True
    bearer = _extract_bearer_token(request.headers.get("authorization", ""))
    if bearer == config.api_key:
        return True
    # 鉴权失败：记录两种 header 的实际前缀（掩码）便于排查。
    # 常见根因：Claude Code 本地 OAuth credentials 优先于 ANTHROPIC_API_KEY，
    # 导致发出的 Bearer token 是 OAuth access token 而非用户配置的 key。
    logger.warning(
        "鉴权失败 path=%s x-api-key=%s authorization-bearer=%s expected=%s",
        request.url.path,
        _mask(x_api_key),
        _mask(bearer or ""),
        _mask(config.api_key),
    )
    return False


async def _check_api_key(request: Request):
    """OpenAI 风格端点鉴权（接受 Bearer 或 x-api-key）。"""
    if not _request_authorized(request):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "无效的 API 密钥",
                    "type": "authentication_error",
                    "param": None,
                    "code": 401,
                }
            },
        )


async def _check_anthropic_api_key(request: Request):
    """Anthropic 风格端点鉴权（接受 x-api-key 或 Bearer），错误体走 Anthropic 形状。"""
    if not _request_authorized(request):
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {"type": "authentication_error", "message": "无效的 API 密钥"},
            },
        )


def _ensure_router_ready():
    """检查路由器是否就绪，未就绪则返回 503。"""
    if router is None:
        raise HTTPException(status_code=503, detail="代理未就绪")


def _extract_bearer_token(auth_header: str) -> str | None:
    """从 Authorization 头提取 Bearer token。

    RFC 7235：scheme 名称大小写不敏感；顺带容忍多空格 / tab 分隔符。
    无效格式返回 None。
    """
    parts = auth_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


@app.get("/v1/models")
async def list_models(request: Request):
    """列出可用模型（三生态：OpenAI / OpenRouter / Anthropic 字段共存，响应同时带 Anthropic 分页 first_id/last_id/has_more）。

    按入站端口选择 provider，MiMo 端口返回 MiMo 专属模型列表，DeepSeek 端口返回 DeepSeek 列表。
    """
    await _check_api_key(request)
    _ensure_router_ready()
    provider, _, port, _ = _binding_for_request(request)
    # pool 配置时列出池内 provider 家族并集
    pool_providers = None
    binding = config.binding_for_port(port) if (config and port is not None) else None
    if binding is not None and binding.model_pool:
        seen: set[str] = set()
        pool_providers = []
        for entry in binding.model_pool:
            p = config.providers.get(entry.provider)
            if p is not None and p.name not in seen:
                pool_providers.append(p)
                seen.add(p.name)
        # home provider 作为排序锚点
        provider = config.provider_for_port(port)
    return await router.list_models(provider=provider, pool_providers=pool_providers)


@app.get("/health")
async def health():
    """健康检查端点。"""
    result: dict = {
        "status": "ok",
        "deepseek_api_key_set": bool(config and config.deepseek.api_key),
        "optimization_enabled": bool(config and config.optimization.enabled),
    }
    if config and router:
        result["flash_upgrade_enabled"] = config.flash_upgrade.enabled
        result["router_type"] = config.flash_upgrade.router_type
        result["writing_basket_kind"] = config.optimization.writing_basket_kind
        # 路由器内部计数器从公开方法读，避免穿透私有属性
        result.update(router.health_snapshot())
    return result


def _binding_for_request(request: Request):
    """按入站端口返回 (provider, sampling_profile, port, selected_model)。

    端口未配置返回 (None, None, None, None)。

    若该 port 配置了 model_pool（writing-port 加权模型桶）：逐请求加权随机选一个
    (provider, model)，provider 覆盖单一绑定、selected_model 为选中的模型 ID
    （供端点覆盖 body["model"]）。无 pool 时 selected_model 为 None。
    """
    if config is None:
        return None, None, None, None
    server = request.scope.get("server")
    port = server[1] if server else None
    if port is None:
        return None, None, None, None
    provider = config.provider_for_port(port)
    sampling = config.sampling_profile_for_port(port)
    selected_model = None
    binding = config.binding_for_port(port)
    if binding is not None and binding.model_pool:
        from .pool import select_pool_target
        provider, selected_model = select_pool_target(binding, config)
        logger.info(
            "[pool] port=%s 加权命中 provider=%s model=%s（池大小=%d）",
            port, provider.name if provider else None, selected_model,
            len(binding.model_pool),
        )
    else:
        logger.info(
            "[pool] port=%s 无 model_pool → 单一 provider=%s（binding=%s）",
            port, provider.name if provider else None,
            "存在" if binding is not None else "缺失",
        )
    return provider, sampling, port, selected_model


def _strip_telemetry_if_enabled(body: Dict[str, Any]) -> None:
    """按 optimization.strip_client_telemetry 配置剥离 user/system 消息中的
    x-anthropic-* telemetry header。在 _maybe_redirect_provider 之前调用，
    确保 RedirectTracker.conversation_fingerprint 看到稳定的首条 user 内容
    （Claude Code 2.1.42+ 注入的 telemetry 含 session hash，每会话变化，
    会让 fingerprint 在首轮就不稳定，破坏 persist 窗口）。

    幂等操作（regex sub），prepare_request step 0c 仍保留同一调用作为
    fallback——便于直接调 router.prepare_request 的测试场景。
    """
    if config is None:
        return
    if not (config.optimization.enabled and config.optimization.strip_client_telemetry):
        return
    from .optimization.strip_telemetry import strip_telemetry_from_messages
    messages = body.get("messages")
    if isinstance(messages, list):
        strip_telemetry_from_messages(messages)


def _maybe_redirect_provider(body, provider):
    """检测 user 消息中的标签或 persist 窗口，必要时覆盖 provider。

    sampling profile 不随重定向变化——标签是"换 provider"而非"换写作风格"，
    入站 port 的 profile 含义对用户更稳定。返回（可能被覆盖的）provider。

    前提：调用方必须先调 _strip_telemetry_if_enabled(body)，否则
    RedirectTracker.conversation_fingerprint 会包含 session-变化的
    telemetry header，让首轮 fingerprint 不稳定。
    """
    if provider is None or config is None or router is None:
        return provider
    from .cross_consult import resolve_redirect
    redirected = resolve_redirect(
        body,
        source_provider=provider,
        config=config,
        tracker=router._redirect_tracker,
    )
    return redirected if redirected is not None else provider


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """聊天补全端点（完全 OpenAI 兼容）。

    根据入站端口选择采样 profile 并强制覆盖 body 中的 4 个采样参数
    （temperature / top_p / presence_penalty / frequency_penalty）。
    """
    await _check_api_key(request)
    _ensure_router_ready()

    body: Dict[str, Any] = await request.json()
    provider, sampling, port, selected_model = _binding_for_request(request)
    # pool 选中的模型覆盖客户端请求的 model（逐请求重掷）
    if selected_model is not None:
        body["model"] = selected_model
    # telemetry 剥离必须先于 redirect/prepare_request，否则两个 tracker 的
    # conversation_fingerprint 会包含 session-变化的 header → persist 窗口失稳
    _strip_telemetry_if_enabled(body)
    pre_redirect_provider = provider
    provider = _maybe_redirect_provider(body, provider)
    # redirect 切换 provider 后，保持 pool 选中的 tier（flash/pro）映射到新 provider，
    # 维持"pro 起始 → pin 在 pro"不变式（见 pool.reconcile_redirected_pool_model）
    if selected_model is not None:
        from .pool import reconcile_redirected_pool_model
        body["model"] = reconcile_redirected_pool_model(
            selected_model, pre_redirect_provider, provider,
        )
    body = await router.prepare_request(
        body, sampling_profile=sampling, provider=provider, port=port,
    )
    is_stream = body.get("stream", False)

    if is_stream:
        return StreamingResponse(
            router.chat_completions_stream(body, provider=provider),
            media_type="text/event-stream",
        )

    try:
        result = await router.chat_completions(body, provider=provider)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("请求处理异常: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"内部错误: {str(e)}", "type": "api_error",
                              "param": None, "code": 500}},
        ) from e


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API 兼容端点。

    把 Anthropic 请求体翻译成 OpenAI 格式，复用现有 router pipeline
    （含模型名规范化、reasoning_content 自愈、采样 profile、skills 优化），
    然后把响应/SSE 流翻译回 Anthropic 格式。
    """
    await _check_anthropic_api_key(request)
    _ensure_router_ready()

    from .compatibility.anthropic_translator import (
        claude_request_to_openai,
        openai_response_to_claude,
        openai_stream_to_claude,
    )

    anthropic_body: Dict[str, Any] = await request.json()
    requested_model = anthropic_body.get("model", "")

    openai_body = claude_request_to_openai(anthropic_body)
    provider, sampling, port, selected_model = _binding_for_request(request)
    # pool 选中的模型覆盖客户端请求的 model（逐请求重掷）
    if selected_model is not None:
        openai_body["model"] = selected_model
    # telemetry 剥离 + cross_consult 标签重定向（顺序同 OpenAI 端点）
    _strip_telemetry_if_enabled(openai_body)
    pre_redirect_provider = provider
    provider = _maybe_redirect_provider(openai_body, provider)
    # redirect 切换 provider 后保持 pool tier（同 OpenAI 端点）
    if selected_model is not None:
        from .pool import reconcile_redirected_pool_model
        openai_body["model"] = reconcile_redirected_pool_model(
            selected_model, pre_redirect_provider, provider,
        )
    openai_body = await router.prepare_request(
        openai_body, sampling_profile=sampling, provider=provider, port=port,
    )
    is_stream = openai_body.get("stream", False)

    if is_stream:
        async def _claude_sse():
            async for event in openai_stream_to_claude(
                router.iter_chat_chunks(openai_body, provider=provider),
                requested_model=requested_model,
            ):
                yield event
        return StreamingResponse(_claude_sse(), media_type="text/event-stream")

    try:
        openai_result = await router.chat_completions(openai_body, provider=provider)
        claude_result = openai_response_to_claude(
            openai_result, requested_model=requested_model,
        )
        return JSONResponse(content=claude_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Anthropic 请求处理异常: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {"type": "api_error", "message": f"内部错误: {str(e)}"},
            },
        ) from e
