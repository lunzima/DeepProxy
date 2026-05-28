"""跨 provider 共享的请求/响应规范化工具。

这里只放与 provider 无关的纯格式清理。任何与 DeepSeek/MiMo 特定模型名、
特定字段名相关的逻辑都放在各自的 *_fixes.py。
"""
from __future__ import annotations

from typing import Any, Dict


def sanitize_stream_options(body: Dict[str, Any]) -> Dict[str, Any]:
    """清理流式响应选项。

    只删空 stream_options（避免传 ``{}`` 触发部分 SDK 校验问题）。
    """
    body = dict(body)
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict) and not stream_options:
        body.pop("stream_options", None)
    return body


def has_tools(body: dict) -> bool:
    """检查请求体是否携带 tools 或 tool_choice。"""
    return bool(body.get("tools") or body.get("tool_choice"))
