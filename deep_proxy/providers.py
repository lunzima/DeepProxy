"""Provider 抽象与端口绑定配置。

provider 之间是对等关系，按 port 分流。配置层不使用 primary/secondary 层级。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Provider(BaseModel):
    """上游 provider 配置。每个 provider 独立持有 api/模型/协议差异参数。"""

    name: str = Field(description="provider 标识，对应 ports[].provider 引用")
    api_base: str = Field(description="上游 API base URL")
    api_key: str = Field(description="上游 API key")
    litellm_prefix: str = Field(description="LiteLLM provider 前缀，如 deepseek/ 或 openai/")
    flash_model: str = Field(description="默认 / flash 档模型 ID")
    pro_model: str = Field(description="升格后的 pro 档模型 ID")

    legacy_aliases: dict[str, dict] = Field(
        default_factory=dict,
        description="客户端历史模型名映射，如 deepseek-chat → flash + thinking 配置",
    )

    has_reasoning_content: bool = Field(
        default=True,
        description="上游响应是否包含 reasoning_content 字段",
    )
    has_thinking_param: bool = Field(
        default=True,
        description="上游是否接受 thinking 协议字段",
    )
    reasoning_effort_field: str = Field(
        default="thinking.reasoning_effort",
        description="reasoning_effort 在请求体中的位置（点号路径）",
    )
    reasoning_effort_value: str = Field(
        default="max",
        description="reasoning_effort 注入的取值",
    )
    thinking_disable_payload: dict = Field(
        default_factory=lambda: {"thinking": {"type": "disabled"}},
        description="禁用思考的 payload 模板",
    )

    allowed_extra_params: list[str] = Field(
        default_factory=list,
        description="LiteLLM 严格校验时需放行的 provider 特有参数（非 OpenAI 标准）。"
                    "例如 MiMo 走 openai/ prefix 但接受 reasoning_effort / thinking。"
                    "为空时不传 allowed_openai_params 给 LiteLLM。",
    )

    max_output_tokens: int = Field(default=384000, gt=0)
    context_window: int = Field(default=1000000, gt=0)


class PortBinding(BaseModel):
    """单个监听端口的绑定：provider + sampling profile。"""

    port: int = Field(ge=1024, le=65535)
    provider: str = Field(description="必须匹配 providers 字典中的某个 key")
    sampling: Literal["precise", "creative"] = Field(
        description="precise → PreciseSamplingConfig；creative → CreativeSamplingConfig",
    )
