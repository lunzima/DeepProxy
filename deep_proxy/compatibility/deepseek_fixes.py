"""DeepSeek API 兼容性修复集合。

包含以下方面的兼容性修复：
1. 模型名称规范化（V4 原生 / 别名映射 / 未知兜底）
2. 默认 thinking.type 推断（legacy alias 隐含 / 未知模型默认 enabled）
3. stream_options 清理（移除空 dict）"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 老模型 ID（2026-07-24 弃用），仅作为 V4 别名保留兼容
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

# V4 系列模型（2026-04 发布）
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

# 老模型 → V4 的默认别名映射（官方文档：2026-07-24 弃用前的兼容层）
# `deepseek-chat`     → deepseek-v4-flash + thinking=disabled
# `deepseek-reasoner` → deepseek-v4-flash + thinking=enabled
DEFAULT_V4_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
}

# 别名隐含的 thinking.type（仅当客户端未显式提供 thinking 时应用）
LEGACY_ALIAS_THINKING = {
    "deepseek-chat": "disabled",
    "deepseek-reasoner": "enabled",
}


def default_thinking_type(raw_model: str) -> Optional[str]:
    """返回模型名隐含的默认 thinking.type；未命中返回 None。

    - 空 / V4 原生名 / 第三方 provider 前缀（含 `/` 非 deepseek/）：返回 None，
      由 router 的 V4 检测路径或客户端显式 thinking 处理
    - deepseek-chat / deepseek-reasoner：按 LEGACY_ALIAS_THINKING
    - 其他未知名（claude-*、gpt-*、误拼写……）：默认 enabled，与
      `normalize_model_name` 的"未知名兜底到 v4-flash"语义对齐
    """
    if not raw_model:
        return None
    if raw_model in V4_MODELS:
        return None
    if "/" in raw_model and not raw_model.startswith("deepseek/"):
        return None
    if raw_model in LEGACY_ALIAS_THINKING:
        return LEGACY_ALIAS_THINKING[raw_model]
    return "enabled"

def normalize_model_name(model: str, model_routes: Optional[List[Dict]] = None) -> str:
    """规范化模型名称为 LiteLLM 兼容格式。

    优先级：
      1. 空名 → 原样返回
      2. V4 新模型（deepseek-v4-flash/pro）— 直接使用
      3. 含 `/` 提供商前缀且非 deepseek 系列 — 透传给 LiteLLM 自行路由
      4. 用户自定义 model_routes — 显式映射优先
      5. 老 deepseek-chat / deepseek-reasoner 别名 → V4
      6. 任何其他未知名（含 claude-*、gpt-*、误拼写等）→ deepseek-v4-flash 兜底
         （单用户玩具代理上游只有 DeepSeek，未知名直接落到 v4-flash 比 502 友好）
    """
    if not model:
        return model

    if model in V4_MODELS:
        return model

    if "/" in model and not model.startswith("deepseek/"):
        return model

    if model_routes:
        for route in model_routes:
            if route["model_name"] == model:
                return route.get("provider_model", model)

    if model in DEFAULT_V4_ALIASES:
        mapped = DEFAULT_V4_ALIASES[model]
        logger.info("模型别名映射: %s → %s", model, mapped)
        return mapped

    # 未知模型兜底（包含 claude-*、gpt-*、其他厂商命名）→ V4 flash
    logger.info("未知模型 %s 兜底映射到 deepseek-v4-flash", model)
    return "deepseek-v4-flash"


def is_v4_model(model: str) -> bool:
    """精确判断是否为 DeepSeek V4 系列模型。

    使用已知 V4 模型列表进行精确匹配，避免子串误判。
    """
    return model in V4_MODELS or model in DEFAULT_V4_ALIASES


def sanitize_stream_options(body: Dict[str, Any]) -> Dict[str, Any]:
    """清理流式响应选项。

    DeepSeek V4 支持 `stream_options.include_usage`，无需特殊处理。
    本函数只删空 stream_options（避免传 `{}` 触发部分 SDK 校验问题）。
    """
    body = dict(body)
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict) and not stream_options:
        body.pop("stream_options", None)
    return body


