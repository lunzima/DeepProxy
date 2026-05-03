"""DeepSeek API 兼容性修复函数。

包含以下方面的兼容性修复：
1. 模型名称规范化（V4 原生 / 别名映射 / 未知兜底）
2. 默认 thinking.type 推断（legacy alias 隐含 / 未知模型默认 enabled）
3. stream_options 清理（移除空 dict）

数据定义已迁移至独立模块：
  - ``deep_proxy.deepseek_models`` — DeepSeek 官方模型 ID 与别名
  - ``deep_proxy.clone_models`` — 仿冒模型别名
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..clone_models import CLONE_MODEL_ALIASES
from ..deepseek_models import (
    DEFAULT_V4_ALIASES,
    LEGACY_ALIAS_THINKING,
    V4_MODELS,
    v4_model_full_set,
)

logger = logging.getLogger(__name__)


def default_thinking_type(raw_model: str) -> Optional[str]:
    """返回模型名隐含的默认 thinking.type；未命中返回 None。

    - 空 / V4 原生名 / DEFAULT_V4_ALIASES（含 [1m] 变体）/ 第三方 provider 前缀
      （含 ``/`` 非 deepseek/）：返回 None，由 router 的 V4 检测路径或客户端显式
      thinking 处理（服务端默认 enabled）
    - deepseek-chat / deepseek-reasoner：按 LEGACY_ALIAS_THINKING
      （chat=disabled 是该别名的"非思考模式"语义；reasoner=enabled）
    - 其他未知名（claude-*、gpt-*、误拼写……）：默认 enabled，与
      ``normalize_model_name`` 的"未知名兜底到 v4-flash"语义对齐

    政策：除显式 thinking={"type":"disabled"} 与 deepseek-chat 别名外，一律默认 enabled。
    """
    if not raw_model:
        return None
    if raw_model in V4_MODELS:
        return None
    if raw_model in DEFAULT_V4_ALIASES and raw_model not in LEGACY_ALIAS_THINKING:
        return None
    if raw_model in CLONE_MODEL_ALIASES:
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
      3. 含 ``/`` 提供商前缀且非 deepseek 系列 — 透传给 LiteLLM 自行路由
      4. 用户自定义 model_routes — 显式映射优先
      5. 老 deepseek-chat / deepseek-reasoner 别名 → V4
      6. 仿冒模型别名 → 对应 V4 模型
      7. 任何其他未知名 → deepseek-v4-flash 兜底
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

    if model in CLONE_MODEL_ALIASES:
        mapped = CLONE_MODEL_ALIASES[model]
        logger.info("仿冒模型映射: %s → %s", model, mapped)
        return mapped

    # 未知模型兜底（包含 claude-*、gpt-*、其他厂商命名）→ V4 flash
    logger.info("未知模型 %s 兜底映射到 deepseek-v4-flash", model)
    return "deepseek-v4-flash"


def is_v4_model(model: str) -> bool:
    """精确判断是否为 DeepSeek V4 系列模型。

    使用完整全集进行精确匹配，避免子串误判。
    """
    return model in v4_model_full_set()


def sanitize_stream_options(body: Dict[str, Any]) -> Dict[str, Any]:
    """清理流式响应选项。

    DeepSeek V4 支持 ``stream_options.include_usage``，无需特殊处理。
    本函数只删空 stream_options（避免传 ``{}`` 触发部分 SDK 校验问题）。
    """
    body = dict(body)
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict) and not stream_options:
        body.pop("stream_options", None)
    return body


def is_thinking_disabled(thinking: Any) -> bool:
    """检查 thinking 对象是否显式 disabled。"""
    return isinstance(thinking, dict) and thinking.get("type") == "disabled"


def has_tools(body: dict) -> bool:
    """检查请求体是否携带 tools 或 tool_choice。"""
    return bool(body.get("tools") or body.get("tool_choice"))


def ensure_thinking_dict(body: dict) -> dict:
    """确保 body 中 thinking 字段为合法 dict，不存在时初始化为空 dict 并写回。

    返回 thinking 引用供调用方操作（如 setdefault）。
    """
    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        thinking = {}
        body["thinking"] = thinking
    return thinking
