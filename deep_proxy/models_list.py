"""模型列表生成 —— OpenRouter 风格 /v1/models 响应。

职责范围：
  - 模型条目标准化（normalize_model_entry）：OpenRouter 风格字段映射
  - 上游 /v1/models 拉取（fetch_upstream_models）

这些函数不依赖路由器实例状态，仅需要纯参数。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .clone_models import CLONE_MODELS
from .deepseek_models import DEEPSEEK_MODELS, V4_MODELS, V4_MODELS_1M
from .deepseek_pricing import _V4_CONTEXT_WINDOW, _V4_MAX_OUTPUT, model_pricing
from .litellm_client import _to_litellm_api_base

logger = logging.getLogger(__name__)


def normalize_model_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """OpenRouter 风格模型条目（带下游 Code Agent 兼容字段）。

    基础字段 `id` / `object` / `created` / `owned_by` 保留兼容，
    增加 `context_length`（OpenRouter）/`max_completion_tokens`/`pricing`，
    并同步输出 `max_model_len`（vLLM/SGLang/Qwen Code）与 `context_window`（部分 Agent 备用）
    以覆盖整个 OpenAI 兼容生态对上下文长度的探测约定。
    """
    model_id = entry["id"]
    created = int(entry.get("created") or 1700000000)

    # 向上游取真实值；无则用默认常量
    ctx = _V4_CONTEXT_WINDOW
    mxo = _V4_MAX_OUTPUT
    for k in ("context_length", "context_window", "max_model_len"):
        v = entry.get(k)
        if isinstance(v, int) and v > 0:
            ctx = v
            break
    v = entry.get("max_output_tokens") or entry.get("max_completion_tokens")
    if isinstance(v, int) and v > 0:
        mxo = v

    out: Dict[str, Any] = {
        "id": model_id,
        "object": entry.get("object", "model"),
        "created": created,
        "owned_by": entry.get("owned_by", "deepseek"),
        # OpenRouter 核心字段
        "context_length": ctx,
        "max_completion_tokens": mxo,
        # 下游 Code Agent 兼容（vLLM/SGLang/Qwen Code/Continue/Cursor/LM Studio/OpenWebUI）
        "max_model_len": ctx,
        "context_window": ctx,
        "pricing": model_pricing(model_id),
        "description": f"DeepSeek V4 — {ctx:,} context window, up to {mxo:,} output tokens",
    }
    return out


async def fetch_upstream_models(
    api_key: str | None,
    api_base: str,
    http_client: httpx.AsyncClient,
) -> list[Dict[str, Any]]:
    """从 DeepSeek 上游 GET /v1/models 拉取真实模型清单。

    返回 None / [] 时表示上游拉取失败，调用方应退化到本地兜底列表。
    """
    if not api_key:
        return []

    base = _to_litellm_api_base(api_base)
    url = f"{base}/models"

    try:
        resp = await http_client.get(
            url, headers={"authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        return list(data) if isinstance(data, list) else []
    except Exception as e:
        logger.warning("上游 /v1/models 拉取失败，使用本地兜底: %s", e)
        return []


def build_models_list(
    raw: list[Dict[str, Any]],
    *,
    expose_legacy_models: bool = False,
    model_routes: list[Dict[str, Any]] | None = None,
) -> list[Dict[str, Any]]:
    """将原始模型条目列表组装为 OpenRouter 风格的输出列表。

    合并上游列表、[1m] 变体、仿冒模型（clone_models）、老别名（DEEPSEEK_MODELS）、
    自定义 model_routes。已存在的 ID 不会重复。
    """
    if not raw:
        raw = list(V4_MODELS.values())

    raw_1m = list(V4_MODELS_1M.values())

    models = [normalize_model_entry(m) for m in raw if isinstance(m, dict) and m.get("id")]
    seen = {m["id"] for m in models}

    # [1m] 变体
    for m in raw_1m:
        if m["id"] not in seen:
            models.append(normalize_model_entry(m))
            seen.add(m["id"])

    # 仿冒模型
    for m in CLONE_MODELS.values():
        if m["id"] not in seen:
            models.append(normalize_model_entry(m))
            seen.add(m["id"])

    # 老别名（当 expose_legacy_models 启用时）
    if expose_legacy_models:
        for m in DEEPSEEK_MODELS.values():
            if m["id"] not in seen:
                models.append(normalize_model_entry(m))
                seen.add(m["id"])

    # 自定义 model_routes
    if model_routes:
        for route in model_routes:
            model_name = route.get("model_name", "")
            if model_name in seen:
                continue
            models.append(normalize_model_entry({
                "id": model_name,
                "owned_by": "deepseek",
            }))
            seen.add(model_name)

    return models
