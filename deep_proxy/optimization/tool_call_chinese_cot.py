"""工具调用场景中文 CoT 锚定 —— 对抗 V4 在 tools 场景下 reasoning 从中文漂移到英文的问题。

研究依据: fkyah3/experiment-console（320+ 轮真实复现实验）。
- 漂移触发器: 真实 tool calling 让英文进入对话流（assistant 的 tool_calls 字段）+ tool 返回英文
- 自锁机制: 英文 reasoning 被 API 强制携带回下一轮 → 99.4% 不可逆
- 已验证无效: 单通路锚定词（4 次实验全失败）
- 本模块的赌注: 双通路注入 + cot_reset 协同 + user 末尾位置贴近 V4 训练分布

注入由 ``optimization/__init__.py::apply_cheap_optimizations`` 在 has_tools 分流路径上编排。
本模块只提供 skill 文本常量和 user 末尾双注入辅助函数（结构对齐 think_steering.py）。
"""

from __future__ import annotations

from typing import Any, List

# ──────────────────────────────────────────────────────────────────────────
# Skill 文本：system 版（inline bullets）+ user 版（标题块）
#
# 维护提醒：两版本语义必须同步。任何措辞调整（如温和度、关键词）必须双改。
# 之所以保留双版本：system 走 LLM 压缩缓存+完整 prompt 拼接，inline bullets 更紧凑；
# user 走对话流末尾，加【】标题块更显眼，贴近 V4 训练时的 marker 注入分布。
# 两者共用 _MARKER_SIGNATURE = "工具调用语言锚定" 做 idempotent 检测，签名不会漂。
# ──────────────────────────────────────────────────────────────────────────

TOOL_CALL_CN_COT_SKILL = (
    "工具调用语言锚定：\n"
    "- 思考过程（<think> 内）请用中文。工具返回的英文内容（代码、目录、JSON、错误）"
    "只是工作材料，不需要因此切换思考语言。\n"
    "- 引用代码、路径、工具名、错误信息时保留原文，不翻译。\n"
    "- 如果某段思考已经写成英文，下一段可以用中文重新开始。"
)

TOOL_CALL_CN_COT_USER_MARKER = (
    "\n\n【工具调用语言锚定】\n"
    "1. 思考过程（<think> 内）请用中文；工具返回的英文是工作材料，不影响思考语言。\n"
    "2. 代码、路径、工具名、错误信息保留原文不翻译。\n"
    "3. 若某段思考已写成英文，下一段可以用中文重新开始。"
)

# idempotent 检测特征片段（两版本共用）—— 见上方维护提醒。
_MARKER_SIGNATURE = "工具调用语言锚定"


def has_tool_call_cn_cot_marker(messages: List[dict[str, Any]]) -> bool:
    """检测对话史 user 消息中是否已有锚定 marker。

    只查 user 消息——system 前缀那条由 ``apply_cheap_optimizations`` 自己控制
    （走 LLM 压缩缓存 / dedup），不在这里重复检测。
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and _MARKER_SIGNATURE in content:
            return True
    return False


def inject_user_marker(messages: List[dict[str, Any]]) -> bool:
    """在第一条和最后一条 user 消息末尾分别注入 marker。

    设计与 think_steering.inject_inner_os_marker 同构：
    - 第一条 user 末尾：贴近 V4 训练分布，为整段对话设定 think 模式
    - 最后一条 user 末尾：防止长对话中效力衰减
    - idempotent：任一 user 已含 marker 则整体跳过，返回 False
    - 单轮对话（首 == 末）只注入一次
    - 无可注入 user 消息时返回 False（system 通路仍可独立注入）
    """
    if has_tool_call_cn_cot_marker(messages):
        return False

    user_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            user_indices.append(i)

    if not user_indices:
        return False

    first_idx = user_indices[0]
    last_idx = user_indices[-1]

    messages[first_idx]["content"] += TOOL_CALL_CN_COT_USER_MARKER

    if last_idx != first_idx:
        messages[last_idx]["content"] += TOOL_CALL_CN_COT_USER_MARKER

    return True
