"""DeepSeek 官方模型标识符与别名映射。

包含 V4 原生模型、[1m] 后缀变体、老 ID 别名（兼容层）及其默认 thinking 配置。
"""

from __future__ import annotations

# ── 老模型 ID（2026-07-24 弃用），仅作 V4 别名保留兼容 ────
DEEPSEEK_MODELS = {
    "deepseek-chat": {
        "id": "deepseek-chat",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
    },
    "deepseek-reasoner": {
        "id": "deepseek-reasoner",
        "object": "model",
        "created": 1700000001,
        "owned_by": "deepseek",
    },
}

# ── V4 系列模型（2026-04 发布） ────────────────────────────
V4_MODELS = {
    "deepseek-v4-flash": {
        "id": "deepseek-v4-flash",
        "object": "model",
        "created": 1745443200,
        "owned_by": "deepseek",
    },
    "deepseek-v4-pro": {
        "id": "deepseek-v4-pro",
        "object": "model",
        "created": 1745443201,
        "owned_by": "deepseek",
    },
}

# ── [1m] 后缀：1M 上下文窗口模型 ID（从 V4_MODELS 派生） ──
V4_MODELS_1M = {
    "deepseek-v4-flash[1m]": {
        key: (value.replace("deepseek-v4-flash", "deepseek-v4-flash[1m]")
              if key == "id" else value)
        for key, value in V4_MODELS["deepseek-v4-flash"].items()
    },
    "deepseek-v4-pro[1m]": {
        key: (value.replace("deepseek-v4-pro", "deepseek-v4-pro[1m]")
              if key == "id" else value)
        for key, value in V4_MODELS["deepseek-v4-pro"].items()
    },
}

# ── 老模型 → V4 默认别名映射 ──────────────────────────────
# deepseek-chat     → deepseek-v4-flash + thinking=disabled
# deepseek-reasoner → deepseek-v4-flash + thinking=enabled
DEFAULT_V4_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
    # [1m] 后缀变体
    "deepseek-v4-flash[1m]": "deepseek-v4-flash",
    "deepseek-v4-pro[1m]": "deepseek-v4-pro",
}

# ── 别名隐含的 thinking.type ──────────────────────────────
LEGACY_ALIAS_THINKING = {
    "deepseek-chat": "disabled",
    "deepseek-reasoner": "enabled",
}
