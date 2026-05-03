"""DeepProxy FastAPI 应用入口。

暴露兼容 OpenAI API + Anthropic Messages API 格式的端点，将请求路由到 DeepSeek 官方 API。

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
    # 仅首次执行完整 startup + shutdown 路径（含 BERT 模型加载与 router.close()）。
    # 第二个实例命中此分支后只 yield，没有 cleanup —— 这是期望行为：
    # shared 资源由首个实例独占管理，第二个实例不应重复释放。
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

    if not config.deepseek.api_key:
        logger.warning("未设置 DEEPSEEK_API_KEY！请通过环境变量或配置文件设置。")

    logger.info(
        "DeepProxy 启动完成 — 监听 %s:%s (coding/precise) + %s:%s (writing/creative, basket=%s)",
        config.host, config.coding_port,
        config.host, config.writing_port,
        config.optimization.writing_basket_kind,
    )

    yield

    if router:
        await router.close()
    logger.info("DeepProxy 已关闭")


app = FastAPI(
    title="DeepProxy",
    description="提升 DeepSeek 官方 API 兼容性的代理服务器",
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


async def _check_api_key(request: Request):
    """检查 OpenAI 风格 Authorization: Bearer 头。"""
    if config and config.api_key:
        auth = request.headers.get("authorization", "")
        if _extract_bearer_token(auth) != config.api_key:
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
    """Anthropic 客户端用 x-api-key 头；同时也接受 Authorization: Bearer 兼容。"""
    if not (config and config.api_key):
        return
    x_key = request.headers.get("x-api-key", "")
    if x_key == config.api_key:
        return
    auth = request.headers.get("authorization", "")
    if _extract_bearer_token(auth) == config.api_key:
        return
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

    遵循 RFC 7235：scheme 大小写不敏感，且容忍多余空格。
    无效格式返回 None。
    """
    parts = auth_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


@app.get("/v1/models")
async def list_models(request: Request):
    """列出可用模型（OpenAI 兼容格式）。"""
    await _check_api_key(request)
    _ensure_router_ready()
    return await router.list_models()


@app.get("/health")
async def health():
    """健康检查端点（含运维诊断信息）。"""
    payload: Dict[str, Any] = {
        "status": "ok",
        "deepseek_api_key_set": bool(config and config.deepseek.api_key),
        "optimization_enabled": bool(config and config.optimization.enabled),
    }
    if config:
        payload["flash_upgrade_enabled"] = config.flash_upgrade.enabled
        payload["router_type"] = config.flash_upgrade.router_type
        payload["writing_basket_kind"] = config.optimization.writing_basket_kind
    if router:
        payload["reasoning_cache_size"] = len(router._reasoning_cache._cache)
        payload["upgrade_tracker_active"] = router._upgrade_tracker.active_count
        payload["upgrade_throttle_size"] = len(router._upgrade_throttle._state)
        if router._compressor is not None:
            payload["compressor_cache_entries"] = len(router._compressor._mem)
    return payload


def _profile_for_request(request: Request):
    """按入站端口选择采样 profile：
    - coding_port → precise_sampling（高确定性）
    - writing_port → creative_sampling（高多样性；写作篮 creative/general 由
      optimization.writing_basket_kind 在 dynamic_baskets 层切换）
    - 其它端口（罕见，例如直接绑定到非配置端口） → None（无强制覆盖）
    """
    if config is None:
        return None
    server = request.scope.get("server")
    port = server[1] if server else None
    if port == config.coding_port:
        return config.precise_sampling
    if port == config.writing_port:
        return config.creative_sampling
    return None


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """聊天补全端点（完全 OpenAI 兼容）。

    根据入站端口选择采样 profile 并强制覆盖 body 中的 4 个采样参数
    （temperature / top_p / presence_penalty / frequency_penalty）。
    """
    await _check_api_key(request)
    _ensure_router_ready()

    body: Dict[str, Any] = await request.json()
    body = await router.prepare_request(
        body, sampling_profile=_profile_for_request(request),
    )
    is_stream = body.get("stream", False)

    if is_stream:
        return StreamingResponse(
            router.chat_completions_stream(body),
            media_type="text/event-stream",
        )

    try:
        result = await router.chat_completions(body)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("请求处理异常: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"内部错误: {str(e)}",
                    "type": "api_error",
                    "param": None,
                    "code": 500,
                }
            },
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
    openai_body = await router.prepare_request(
        openai_body, sampling_profile=_profile_for_request(request),
    )
    is_stream = openai_body.get("stream", False)

    if is_stream:
        async def _claude_sse():
            # 直接接业务层 dict 流，跳过 OpenAI SSE 协议层
            async for event in openai_stream_to_claude(
                router.iter_chat_chunks(openai_body),
                requested_model=requested_model,
            ):
                yield event
        return StreamingResponse(_claude_sse(), media_type="text/event-stream")

    try:
        openai_result = await router.chat_completions(openai_body)
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
