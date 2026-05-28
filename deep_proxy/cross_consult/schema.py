"""Cross-Consult 工具 schema 与 system prompt 增量构造。

工具描述与提示词使用中性技术语言，遵循 spec §3.2 / §3.3：
- 不出现"主模型 / 副模型 / 备用模型"等层级措辞
- 不叙事化 provider 间关系
"""
from __future__ import annotations

from typing import Any


def build_tool_schema(*, tool_name: str = "cross_consult") -> dict[str, Any]:
    """构造 OpenAI function-calling 风格的工具 schema。"""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "向异分布模型家族请求对当前问题的视角。"
                "本工具由 DeepProxy 执行，不经过客户端文件系统或 shell。\n"
                "问题必须 self-contained——目标模型没有本次会话的上下文。\n"
                "适用：跨领域子任务（写作里的逻辑、编码里的语感）、"
                "寻求二次验证、打破认知惯性。"
            ),
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "明确的问题或子任务，self-contained。",
                    },
                    "context": {
                        "type": "string",
                        "description": "可选背景。如果 question 引用了当前会话中的"
                                       "代码/文本，把片段放在这里。",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": [
                            "second_opinion",
                            "cross_domain_help",
                            "style_check",
                            "logic_check",
                            "other",
                        ],
                        "description": "调用意图，用于 telemetry，不影响行为。",
                    },
                },
            },
        },
    }


def build_system_prompt_addendum(*, tool_name: str, max_calls: int) -> str:
    """构造追加到 system prompt 末尾的简短工具说明。

    该说明不参与 prompt 压缩缓存（含运行时 max_calls，会破坏缓存命中）。
    """
    return (
        f"\n\n[DeepProxy] 你可调用工具 `{tool_name}` 向异家族模型请求第二视角。\n"
        f"适用：跨领域子任务、寻求二次验证、打破认知惯性。\n"
        f"注意：目标模型没有本次会话上下文，question 必须 self-contained，"
        f"需要时把相关片段放在 context 字段。\n"
        f"本次会话最多调用 {max_calls} 次。"
    )
