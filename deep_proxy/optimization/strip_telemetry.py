"""客户端 telemetry header 行剥离。

针对 Claude Code 2.1.42+ 在 system prompt 头部注入 `x-anthropic-billing-header:`
这类含 session hash 的伪 header 行：每次新会话哈希变化破坏 prefix cache。
本模块只动消息体里的伪 header 文本，不动 HTTP 请求的真实 header。

设计：
- 单一来源的共享正则 `_TELEMETRY_HEADER_RE`（compressor.py 通过 import 复用）。
- `strip_telemetry_from_text`：纯函数，行首匹配 `x-anthropic-<name>:` 整行删除。
- `strip_telemetry_from_messages`：就地处理 system 消息 + 首条 user 消息。
- 容错优先于诊断：异常情况静默跳过，不阻塞主请求。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 匹配 Claude Code 等客户端在 system prompt 中嵌入的伪 header 行。
# - ^ + re.MULTILINE：仅行首匹配，避免误删正文中作为引用出现的字符串。
# - re.IGNORECASE：CC 历史版本曾用 X-Anthropic-* / x-anthropic-* 混合大小写。
_TELEMETRY_HEADER_RE = re.compile(
    r"^x-anthropic-[a-z-]+:.*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_telemetry_from_text(text: Any) -> Any:
    """剥离文本中匹配 telemetry header 模式的整行。

    - 非 str 输入原样返回（容错；调用方可能传 None / list 等）
    - 不合并连续空行：删除后留下的空行对下游 LLM 语义无影响
    """
    if not isinstance(text, str):
        return text
    return _TELEMETRY_HEADER_RE.sub("", text)


def _strip_in_place(msg: dict) -> None:
    """就地处理单条消息的 content 字段。"""
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = strip_telemetry_from_text(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = strip_telemetry_from_text(block["text"])
    # 其它类型（数字 / None / 自定义对象）静默跳过


def strip_telemetry_from_messages(messages: Iterable[dict] | None) -> None:
    """就地剥离消息列表中的 telemetry header 行。

    扫描范围：
    - 所有 role=system 消息
    - 第一条 role=user 消息（Claude Code 的注入位置）

    其它后续 user 消息可能合法引用 `x-anthropic-*: ...` 字符串作为示例，不处理。
    """
    if not isinstance(messages, list) or not messages:
        return

    try:
        first_user_seen = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                _strip_in_place(msg)
            elif role == "user" and not first_user_seen:
                _strip_in_place(msg)
                first_user_seen = True
    except Exception as e:
        # 永不阻塞主请求：异常仅记日志，messages 保持原样
        logger.warning("strip_telemetry_from_messages 异常，跳过: %s", e)
