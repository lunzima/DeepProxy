"""CrossConsultConfig pydantic 模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


_DEFAULT_CONSULT_SYSTEM_PROMPT = (
    "你被作为外部顾问从另一对话中召唤。直接回答问题，不寒暄，不复述问题。\n"
    "你没有该对话的上下文，只能依据本次提问中给出的信息作答。\n"
    "如果信息不足以回答，明确说明缺少什么。"
)


class CrossConsultConfig(BaseModel):
    """Cross-Consult 配置。pairs 对称声明 provider 间对偶关系。

    pairs 例：{"deepseek": "mimo", "mimo": "deepseek"}
    设计参考 docs/mimo_integration.md §12。
    """

    enabled: bool = Field(
        default=False,
        description="主开关。默认关闭——用户须显式开启并配置 pairs。",
    )
    tool_name: str = Field(
        default="cross_consult",
        description="工具暴露给 agent 的名字。",
    )
    pairs: dict[str, str] = Field(
        default_factory=dict,
        description="provider 对偶 map（symmetric，由用户在两个方向各声明一次）。"
                    "例：{'deepseek': 'mimo', 'mimo': 'deepseek'}。",
    )
    max_calls_per_request: int = Field(
        default=3, ge=1, le=10,
        description="单次 client 请求内 cross_consult 调用次数上限。",
    )
    call_timeout_seconds: int = Field(
        default=30, ge=1, le=600,
        description="单次 consult 调用超时（秒）。",
    )
    max_input_chars: int = Field(
        default=32000, ge=100,
        description="question + context 合并后的字符上限；超出返回错误 tool_result。",
    )
    max_output_tokens: int = Field(
        default=4096, ge=1,
        description="consult 调用的 max_tokens。",
    )
    consult_system_prompt: str = Field(
        default=_DEFAULT_CONSULT_SYSTEM_PROMPT,
        description="consult 调用时用作 system 消息的提示词。",
    )

    def pair_for(self, source_provider: str) -> str | None:
        """返回 source_provider 的对偶名；未开启或未配置返回 None。"""
        if not self.enabled:
            return None
        return self.pairs.get(source_provider)
