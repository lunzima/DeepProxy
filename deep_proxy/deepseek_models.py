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

# ── V4 系列模型常量（单一数据源，避免各处硬编码字符串） ────
V4_FLASH = "deepseek-v4-flash"
V4_PRO = "deepseek-v4-pro"

# ── V4 系列模型（2026-04 发布） ────────────────────────────
V4_MODELS = {
    V4_FLASH: {
        "id": V4_FLASH,
        "object": "model",
        "created": 1745443200,
        "owned_by": "deepseek",
    },
    V4_PRO: {
        "id": V4_PRO,
        "object": "model",
        "created": 1745443201,
        "owned_by": "deepseek",
    },
}

# ── [1m] 后缀：1M 上下文窗口模型 ID（从 V4_MODELS 派生） ──
V4_MODELS_1M = {
    f"{V4_FLASH}[1m]": {
        key: (value.replace(V4_FLASH, f"{V4_FLASH}[1m]")
              if key == "id" else value)
        for key, value in V4_MODELS[V4_FLASH].items()
    },
    f"{V4_PRO}[1m]": {
        key: (value.replace(V4_PRO, f"{V4_PRO}[1m]")
              if key == "id" else value)
        for key, value in V4_MODELS[V4_PRO].items()
    },
}

# ── 老模型 → V4 默认别名映射 ──────────────────────────────
# deepseek-chat     → deepseek-v4-flash + thinking=disabled（"非思考模式"显式别名）
# deepseek-reasoner → deepseek-v4-flash + thinking=enabled
DEFAULT_V4_ALIASES = {
    "deepseek-chat": V4_FLASH,
    "deepseek-reasoner": V4_FLASH,
    # [1m] 后缀变体
    f"{V4_FLASH}[1m]": V4_FLASH,
    f"{V4_PRO}[1m]": V4_PRO,
}

# ── 别名隐含的 thinking.type ──────────────────────────────
# 政策：除非客户端显式 thinking={"type":"disabled"} 或选用 deepseek-chat 这类
# "非思考模式"专属别名，否则默认 enabled。其它别名 / V4 原生 / 未知名一律 enabled。
LEGACY_ALIAS_THINKING = {
    "deepseek-chat": "disabled",
    "deepseek-reasoner": "enabled",
}

# ── V4 模型全集（用于快速成员检测） ──────────────────────
# 合并 V4_MODELS 的键、DEFAULT_V4_ALIASES 的键（含 [1m] 变体）、仿冒别名
# 注：仿冒别名需延迟导入（clone_models 依赖 deepseek_models 的 V4_FLASH/PRO）
_V4_FULL_SET: set | None = None


def _build_v4_full_set() -> set:
    """延迟组装 V4 模型全集，避免循环导入。"""
    from .clone_models import CLONE_MODEL_ALIASES
    return set(V4_MODELS.keys()) | set(DEFAULT_V4_ALIASES.keys()) | set(CLONE_MODEL_ALIASES.keys())


def v4_model_full_set() -> set:
    """返回 V4 模型名全集。"""
    global _V4_FULL_SET
    if _V4_FULL_SET is None:
        _V4_FULL_SET = _build_v4_full_set()
    return _V4_FULL_SET
