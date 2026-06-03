"""CrossConsultConfig pydantic 模型。"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, PrivateAttr


_DEFAULT_CONSULT_SYSTEM_PROMPT = (
    "你被作为外部顾问从另一对话中召唤。直接回答问题，不寒暄，不复述问题。\n"
    "你没有该对话的上下文，只能依据本次提问中给出的信息作答。\n"
    "如果信息不足以回答，明确说明缺少什么。"
)


_DEFAULT_REDIRECT_TAG = r"\[本轮对话使用不同家族的大语言模型\]"


class CrossConsultConfig(BaseModel):
    """Cross-Consult 配置。pairs 对称声明 provider 间对偶关系。

    pairs 例：{"deepseek": "mimo", "mimo": "deepseek"}
    设计参考 docs/mimo_integration.md §12 / §12.11。
    """

    enabled: bool = Field(
        default=True,
        description="主开关。默认开启——配置 pairs 后两个机制都激活。",
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
        default=60, ge=1, le=600,
        description="流式 chunk 间最大空闲秒数（inter-chunk idle）。首 chunk 之后，"
                    "相邻 chunk 间隔超过该值才视为 mid-stream hang。注：首 chunk 的 "
                    "prefill/TTFT 由 first_chunk_timeout_seconds 单独管辖。",
    )
    first_chunk_timeout_seconds: int = Field(
        default=120, ge=1, le=600,
        description="等待首个 chunk（prefill / TTFT + 推理预热）的上限秒数。"
                    "cross_consult 重发携带最大上下文（原对话 + 注入的 tool_result），"
                    "其 time-to-first-chunk 可能远超 inter-chunk idle 预算；首 chunk "
                    "因此单独给更宽预算，避免健康上游在预填充阶段被误杀。",
    )
    stream_heartbeat_seconds: int = Field(
        default=10, ge=1, le=120,
        description="客户端真流式下，静默间隙（consult 执行 / 重发 prefill）期间发送 "
                    "SSE keep-alive 注释帧的间隔秒数。须显著小于客户端 idle-read 超时。",
    )
    # 注：consult 的输入/输出不设武断上限。真正的约束是 target provider 自身的
    # context_window（输入）与 max_output_tokens（输出）——executor 用 provider 的
    # 输出上限做 max_tokens，输入超长由 provider 自然报错并以 tool_result 返还 agent。
    consult_system_prompt: str = Field(
        default=_DEFAULT_CONSULT_SYSTEM_PROMPT,
        description="consult 调用时用作 system 消息的提示词。",
    )

    # --- 标签重定向（§12.11）---
    redirect_enabled: bool = Field(
        default=True,
        description="user 消息标签触发的整轮重定向开关。"
                    "命中 redirect_tag_pattern 时把请求重路由到异家族 provider。",
    )
    redirect_persist_turns: int = Field(
        default=2, ge=0, le=20,
        description="标签触发后，后续 N 轮额外保持重定向（语义同 flash_upgrade.persist_turns 但独立计数）。",
    )
    redirect_tag_pattern: str = Field(
        default=_DEFAULT_REDIRECT_TAG,
        description="user 消息中触发重定向的正则。默认匹配字面 [本轮对话使用不同家族的大语言模型]，"
                    "适度宽容（允许周围空白、全/半角方括号等变体可由用户自定义）。",
    )
    awareness_enabled: bool = Field(
        default=True,
        description="是否在 system prompt 中注入双家族状态披露。关掉只剩工具 schema 自身的简短说明。",
    )

    # 编译后的正则（PrivateAttr 避免 pydantic 校验/序列化 / 不进 dict）
    _compiled_redirect: re.Pattern[str] | None = PrivateAttr(default=None)

    def pair_for(self, source_provider: str) -> str | None:
        """返回 source_provider 的对偶名；未开启或未配置返回 None。"""
        if not self.enabled:
            return None
        return self.pairs.get(source_provider)

    def compiled_redirect_pattern(self) -> re.Pattern[str]:
        """返回编译后的标签正则（lazy + 缓存）。"""
        if self._compiled_redirect is None:
            self._compiled_redirect = re.compile(self.redirect_tag_pattern)
        return self._compiled_redirect
