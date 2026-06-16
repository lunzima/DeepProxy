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

    allowed_extra_params: list[str] = Field(
        default_factory=list,
        description="LiteLLM 严格校验时需放行的 provider 特有参数（非 OpenAI 标准）。"
                    "例如 MiMo 走 openai/ prefix 但接受 reasoning_effort / thinking。"
                    "为空时不传 allowed_openai_params 给 LiteLLM。",
    )

    max_output_tokens: int = Field(default=384000, gt=0)
    context_window: int = Field(default=1000000, gt=0)


class PoolEntry(BaseModel):
    """加权模型桶的单个条目：(provider, model) + 相对权重。"""

    provider: str = Field(description="必须匹配 providers 字典中的某个 key")
    model: str = Field(description="必须等于该 provider 的 flash_model 或 pro_model")
    weight: float = Field(default=1.0, gt=0, description="加权随机的相对权重（> 0）")


class PortBinding(BaseModel):
    """单个监听端口的绑定：provider + sampling profile（+ 可选加权模型桶）。"""

    port: int = Field(ge=1024, le=65535)
    provider: str = Field(description="必须匹配 providers 字典中的某个 key")
    sampling: Literal["precise", "creative"] = Field(
        description="precise → PreciseSamplingConfig；creative → CreativeSamplingConfig",
    )
    system_prompt: str | None = Field(
        default=None,
        description="可选的端口级角色扮演 system prompt（1-3 句，描述角色身份与风格）。"
                    "在全部 skills / 压缩之后注入到 system 消息的最开头。",
    )
    model_pool: list[PoolEntry] | None = Field(
        default=None,
        description="加权随机模型桶。给定时该 port 逐请求从池中选 (provider, model)，"
                    "覆盖单一 provider 路由；provider 字段仍作为 home/兜底"
                    "（/v1/models 默认）。条目的 provider/model 合法性由 "
                    "ProxyConfig 的 model_validator 校验（providers 字典就绪后）。",
    )
    system_prompt: str | None = Field(
        default=None,
        description="该 port 的 persona 系统提示词。给定时逐请求以 system 消息注入到 "
                    "messages 最前（参与压缩/skills 流水线）。用于按 port 绑定写作/编码人格。",
    )
