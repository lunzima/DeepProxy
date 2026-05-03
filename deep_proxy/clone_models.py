"""仿冒（clone）模型别名：客户端用这些名字请求时内部路由到 DeepSeek V4。

数据源：
  - OpenRouter API /v1/models（实际路由名，带 dot / 不带日期后缀）
  - Anthropic 官方文档（claude-*-4-* 格式的官方 API ID + alias）
  - Google 官方文档 / OpenRouter（gemini-*-preview 格式）

入选标准：仅保留能力不低于 deepseek-v4-flash 的模型（可用于严肃编程）。
排除轻量级变体：haiku / mini / nano / lite / flash-lite 等弱于 v4-flash 的不加入。

分层策略：
  *pro / opus / codex → deepseek-v4-pro（顶级 / 代码能力）
  sonnet / 无后缀 → deepseek-v4-flash（均衡 / 快速）
"""

from __future__ import annotations

from .deepseek_models import V4_FLASH, V4_PRO

CLONE_MODEL_ALIASES = {
    # ── Anthropic Claude（OpenRouter 路由名 + 官方 API ID） ──
    # OpenRouter 使用 dot 分隔符
    "claude-opus-4.7": V4_PRO,
    "claude-opus-4.6": V4_PRO,
    "claude-opus-latest": V4_PRO,
    "claude-sonnet-4.6": V4_FLASH,
    # Anthropic 官方 API ID 使用 dash 分隔符（与 dot 共存，客户端两者皆用）
    "claude-opus-4-7": V4_PRO,
    "claude-opus-4-6": V4_PRO,
    "claude-sonnet-4-6": V4_FLASH,

    # ── OpenAI（OpenRouter 路由名） ─────────────────────────
    "gpt-5.5-pro": V4_PRO,
    "gpt-5.5": V4_FLASH,
    "gpt-5.4-pro": V4_PRO,
    "gpt-5.4": V4_FLASH,

    # ── Google Gemini（OpenRouter 路由名） ──────────────────
    "gemini-3.1-pro-preview": V4_PRO,
}

# 仿冒模型在 /v1/models 中的展示条目（normalize_model_entry 会补齐三生态字段）
CLONE_MODELS = {
    name: {
        "id": name,
        "object": "model",
        "created": 1745443200,
        "owned_by": "deepseek",
    }
    for name in CLONE_MODEL_ALIASES
}
