"""V4 <think> 引导模块 —— 通过第一轮 user 消息末尾注入 marker，
引导 V4 的推理过程（<think>）进入角色沉浸模式。

参考: victorchen96/deepseek_v4_rolepaly_instruct
适用: DeepSeek V4 (20260424), deepseek-v4-flash / deepseek-v4-pro
"""

from __future__ import annotations

from typing import Any, List

# 角色沉浸 marker：引导 <think> 以角色第一人称进行内心独白
INNER_OS_MARKER = (
    "\n\n【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：\n"
    "1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"
    '"（心想：……）"或"(内心OS：……)"\n'
    "2. 用第一人称描写角色的内心感受，例如"
    '"我心想""我觉得""我暗自"等\n'
    "3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"
)

# marker 特征片段：用于 idempotent 检测（只取开头即可唯一识别）
_MARKER_SIGNATURE = "【角色沉浸要求】"


def has_inner_os_marker(messages: List[dict[str, Any]]) -> bool:
    """检测对话史中是否已有角色沉浸 marker。

    遍历所有 user 消息内容，检查是否包含 marker 特征片段。
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and _MARKER_SIGNATURE in content:
            return True
    return False


def inject_inner_os_marker(messages: List[dict[str, Any]]) -> bool:
    """在最后一条 user 消息末尾注入角色沉浸 marker。

    - idempotent：若已有 marker 则跳过，返回 False
    - 若最后一条消息非 user 或无 content 字符串，返回 False
    - 注入成功返回 True

    marker 注入位置为 user 消息末尾（非 system prompt），与 V4 训练时的
    注入位置一致，效果最稳定。后续轮次正常追加消息，marker 在对话史中
    持续发挥效力。
    """
    if has_inner_os_marker(messages):
        return False

    # 找最后一条 user 消息
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        msg["content"] = content + INNER_OS_MARKER
        return True

    return False
