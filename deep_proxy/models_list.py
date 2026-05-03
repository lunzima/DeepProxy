"""模型列表生成 —— /v1/models 响应（OpenAI / OpenRouter / Anthropic 三生态字段共存）。

职责范围：
  - 模型条目标准化（normalize_model_entry）：单条目同时输出 OpenAI/OpenRouter
    （id/object/created/context_length/pricing 等）与 Anthropic
    （type/display_name/created_at/max_input_tokens/max_tokens）字段。
    上下文长度 / 输出上限存在多字段冗余（同值多键），完整字段→消费方表见
    normalize_model_entry 的 docstring。故意不输出 `capabilities`
    （避免谎报代理未实现的 Anthropic context-management beta 行为）。
  - 上游 /v1/models 拉取（fetch_upstream_models）
  - 列表组装（build_models_list）：合并上游 + [1m] 变体 + 仿冒别名 + 老别名 + 自定义路由

这些函数不依赖路由器实例状态，仅需要纯参数。响应顶层 Anthropic 分页字段
（first_id/last_id/has_more）由 router.list_models 在调用方拼接。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict

import httpx

from .clone_models import CLONE_MODELS
from .deepseek_models import DEEPSEEK_MODELS, V4_MODELS, V4_MODELS_1M
from .deepseek_pricing import _V4_CONTEXT_WINDOW, _V4_MAX_OUTPUT, model_pricing
from .litellm_client import _to_litellm_api_base

logger = logging.getLogger(__name__)


def _epoch_to_iso8601(epoch: int) -> str:
    """epoch 秒 → ISO 8601 UTC（Anthropic 风格 created_at）。"""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── display_name 段表覆盖（避免 .capitalize() 把品牌名/缩写写丑） ──
# 扩展时只加广为人知的厂商/产品段；模糊大小写场景由上游 entry["display_name"] 透传覆盖。
# MappingProxyType 包裹防外部代码意外 mutation（仅读取场景）。
_SEG_OVERRIDES: "MappingProxyType[str, str]" = MappingProxyType({
    "deepseek": "DeepSeek",
    "v4": "V4", "v3": "V3", "v2": "V2",
    "gpt": "GPT", "4o": "4o",
    "claude": "Claude",
    "gemini": "Gemini",
    "qwen": "Qwen",
    "llama": "Llama",
    "mistral": "Mistral",
    "grok": "Grok",
    "openai": "OpenAI", "anthropic": "Anthropic",
})

# 参数量段（72b / 405B / 1.5b）—— 数字 + b/B 结尾，统一大写为业内惯例 "72B"。
_PARAM_COUNT_RE = re.compile(r"^\d+(?:\.\d+)?[bB]$")


def _build_display_name(model_id: str) -> str:
    """`deepseek-v4-flash` → `DeepSeek V4 Flash`；`gpt-4o-mini` → `GPT 4o Mini`；
    `qwen-72b` → `Qwen 72B`；`llama-3.1-405b-instruct` → `Llama 3.1 405B Instruct`。

    `[1m]` 等方括号后缀单独成段保留原样（不被 `.capitalize()` 改写）。
    """
    parts: list[str] = []
    for seg in model_id.replace("[", " [").split("-"):
        if seg.startswith("[") and seg.endswith("]"):
            parts.append(seg)            # 方括号后缀原样保留
            continue
        if _PARAM_COUNT_RE.match(seg):
            parts.append(seg.upper())    # 72b → 72B
            continue
        parts.append(_SEG_OVERRIDES.get(seg.lower(), seg.capitalize()))
    return " ".join(parts)


def normalize_model_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """统一模型条目（同时覆盖 OpenAI / OpenRouter / Anthropic 三种生态字段）。

    上下文长度 / 输出上限的字段冗余表（全部同值，下游各取所需，未知字段静默忽略）：

    | 字段 | 取值 | 主消费方 |
    |------|------|----------|
    | `max_input_tokens`     | ctx | Anthropic SDK / Claude Code / Qwen Code（anthropic provider） |
    | `context_length`       | ctx | OpenRouter SDK / Continue.dev / OpenWebUI |
    | `max_model_len`        | ctx | vLLM / SGLang / Qwen Code（openai provider） |
    | `context_window`       | ctx | Aider 旧版 / 自定义集成 |
    | `max_tokens`           | mxo | Anthropic SDK |
    | `max_completion_tokens`| mxo | OpenAI / OpenRouter |

    上游 entry 优先级：Anthropic 原生字段（`max_input_tokens` / `max_tokens`）排首位，
    因为若上游本身是 Anthropic-shaped 服务，其原生字段是最权威信号。
    """
    model_id = entry["id"]
    created = int(entry.get("created") or 1700000000)

    # 向上游取真实值；无则用默认常量
    # 顺序：Anthropic 原生 > OpenRouter > vLLM/SGLang > 旧 OpenAI 字段
    ctx = _V4_CONTEXT_WINDOW
    mxo = _V4_MAX_OUTPUT
    for k in ("max_input_tokens", "context_length", "max_model_len", "context_window"):
        v = entry.get(k)
        if isinstance(v, int) and v > 0:
            ctx = v
            break
    for k in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
        v = entry.get(k)
        if isinstance(v, int) and v > 0:
            mxo = v
            break

    # display_name：上游显式提供则透传，否则按段表生成
    display_name = entry.get("display_name") or _build_display_name(model_id)

    out: Dict[str, Any] = {
        # ── OpenAI / OpenRouter 字段 ─────────────────────────────────
        "id": model_id,
        "object": entry.get("object", "model"),
        "created": created,
        "owned_by": entry.get("owned_by", "deepseek"),
        "context_length": ctx,
        "max_completion_tokens": mxo,
        "max_model_len": ctx,        # vLLM/SGLang/Qwen Code
        "context_window": ctx,       # 部分 Agent 备用
        "pricing": model_pricing(model_id),
        # ── Anthropic 字段 ──────────────────────────────────────────
        # Anthropic 真实 /v1/models 仅含 {type, id, display_name, created_at}；
        # max_input_tokens/max_tokens 是社区扩展（Qwen Code 等读取），
        # 故意不输出 `capabilities` —— 代理本身不实现 Anthropic context-management
        # beta API，谎报会让 Claude Code/Aider 误启用代理无法兑现的行为。
        "type": "model",
        "display_name": display_name,
        "created_at": _epoch_to_iso8601(created),
        "max_input_tokens": ctx,
        "max_tokens": mxo,
        # ── 共用 ─────────────────────────────────────────────────────
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
    """将原始模型条目列表组装为三生态共存的输出列表。

    合并上游列表、[1m] 变体、仿冒模型（clone_models）、老别名（DEEPSEEK_MODELS）、
    自定义 model_routes。已存在的 ID 不会重复。每个条目通过 normalize_model_entry
    同时携带 OpenAI/OpenRouter 与 Anthropic 两套字段。
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
