"""测试 DeepProxyRouter.prepare_request 端到端管道。"""
from __future__ import annotations

import pytest

from deep_proxy.config import DeepSeekConfig, ProxyConfig
from deep_proxy.router import DeepProxyRouter
from deep_proxy.litellm_client import _ensure_string_content


class TestPrepareRequestChat:
    async def test_reasoner_alias_to_v4_flash_thinking_enabled(self, router: DeepProxyRouter):
        """官方：deepseek-reasoner 是 deepseek-v4-flash 的思考模式别名。"""
        body = {
            "model": "deepseek-reasoner",
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        p = await router.prepare_request(body)
        assert p["model"] == "deepseek-v4-flash"
        # 别名隐含 thinking=enabled
        assert p["thinking"]["type"] == "enabled"
        # V4 全部支持采样参数与 response_format（无特殊剥离）
        assert p["temperature"] == 0.7
        assert "response_format" in p
        assert p["stream_options"] == {"include_usage": True}
        # thinking 已扩充 reasoning_effort=max
        assert p["thinking"]["reasoning_effort"] == "max"

    async def test_chat_alias_to_v4_flash_thinking_disabled(self, router: DeepProxyRouter):
        """官方：deepseek-chat 是 deepseek-v4-flash 的非思考模式别名。"""
        body = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
        }
        p = await router.prepare_request(body)
        assert p["model"] == "deepseek-v4-flash"
        # 别名隐含 thinking=disabled
        assert p["thinking"]["type"] == "disabled"
        # disabled 时不注入 reasoning_effort
        assert "reasoning_effort" not in p["thinking"]

    async def test_v4_default_keeps_api_default_thinking(self, router: DeepProxyRouter):
        """直接用 deepseek-v4-flash 且无 thinking 字段时，代理仅注入 reasoning_effort=max，
        不强制 thinking.type，让 API 默认 enabled 生效。
        """
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        p = await router.prepare_request(body)
        assert p["model"] == "deepseek-v4-flash"
        assert p["thinking"].get("type") != "disabled"
        assert p["thinking"]["reasoning_effort"] == "max"

    async def test_v4_explicit_thinking_disabled_respected(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body)
        assert p["thinking"] == {"type": "disabled"}
        # disabled 时不强行注入 reasoning_effort
        assert "reasoning_effort" not in p["thinking"]

    async def test_v4_explicit_thinking_enabled_keeps_user_reasoning_effort(
        self, router: DeepProxyRouter
    ):
        body = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "enabled", "reasoning_effort": "high"},
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body)
        # 客户端已传值，不被覆盖
        assert p["thinking"]["reasoning_effort"] == "high"


class TestDefaultParamInjection:
    """默认值：roleplay 默认开 → 从配置范围抽 sampling 参数；reasoning_effort=max。"""

    async def test_defaults_injected_when_missing(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body)
        # roleplay 默认开 → 从范围抽样
        rp = router.config.creative_sampling
        assert rp.temperature_min <= p["temperature"] <= rp.temperature_max
        assert rp.top_p_min <= p["top_p"] <= rp.top_p_max
        # reasoning_effort 应在 thinking 对象内部
        assert p["thinking"]["reasoning_effort"] == "max"
        # 不应在顶层
        assert "reasoning_effort" not in p

    async def test_client_values_take_precedence(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "thinking": {"type": "enabled", "reasoning_effort": "high"},
            "temperature": 0.1,
            "top_p": 0.5,
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body)
        assert p["temperature"] == 0.1
        assert p["top_p"] == 0.5
        assert p["thinking"]["reasoning_effort"] == "high"

    async def test_v4_thinking_enabled_keeps_sampling_defaults(self, router: DeepProxyRouter):
        """thinking=enabled 时采样参数不被剥离（V4 全部支持）。"""
        body = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body)
        rp = router.config.creative_sampling
        assert rp.temperature_min <= p["temperature"] <= rp.temperature_max
        assert rp.top_p_min <= p["top_p"] <= rp.top_p_max
        assert p["thinking"]["reasoning_effort"] == "max"

    async def test_reasoner_alias_keeps_sampling_params(self, router: DeepProxyRouter):
        """官方文档未限制 reasoner 别名采样参数；映射到 v4-flash 后仍按 roleplay 抽样。"""
        body = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "x"}],
        }
        p = await router.prepare_request(body)
        assert p["model"] == "deepseek-v4-flash"
        rp = router.config.creative_sampling
        assert rp.temperature_min <= p["temperature"] <= rp.temperature_max
        assert rp.top_p_min <= p["top_p"] <= rp.top_p_max
        assert p["thinking"]["type"] == "enabled"


class TestEnsureStringContent:
    """_ensure_string_content 数组 content 展平。"""

    def test_array_with_text_parts_joined_by_double_newline(self):
        """[{type:text}, {type:text}] → 双换行连接。"""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "world"},
            ]},
        ]
        result = _ensure_string_content(messages)
        assert result[0]["content"] == "Hello\n\nworld"
        # 不应修改原始列表
        assert isinstance(messages[0]["content"], list)

    def test_string_content_passthrough(self):
        """纯字符串 content 原样保留。"""
        messages = [{"role": "user", "content": "Hello world"}]
        result = _ensure_string_content(messages)
        assert result[0]["content"] == "Hello world"
        assert result is not messages  # 新列表

    def test_mixed_plain_string_and_object_parts(self):
        """数组混含纯字符串片段和对象片段。"""
        messages = [
            {"role": "user", "content": [
                "Hello",
                {"type": "text", "text": "world"},
            ]},
        ]
        result = _ensure_string_content(messages)
        assert result[0]["content"] == "Hello\n\nworld"

    def test_non_text_part_replaced_with_placeholder(self):
        """非文本部件（image_url）替换为占位符。"""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/..."}},
            ]},
        ]
        result = _ensure_string_content(messages)
        assert "Describe this image" in result[0]["content"]
        assert "[Unsupported content type: image_url]" in result[0]["content"]

    def test_multiple_messages_mixed_content_types(self):
        """多条消息混合数组与字符串。"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Hi"},
            ]},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = _ensure_string_content(messages)
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["content"] == "Hi"
        assert result[2]["content"] == "Hello!"

    def test_none_content_preserved(self):
        """content 为 None 时不做处理。"""
        messages = [{"role": "assistant", "content": None, "tool_calls": []}]
        result = _ensure_string_content(messages)
        assert result[0]["content"] is None

    def test_unknown_part_type(self):
        """未知部件类型。"""
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": {"data": "..."}},
            ]},
        ]
        result = _ensure_string_content(messages)
        assert result[0]["content"] == "[Unsupported content type: audio]"

    def test_empty_list_content(self):
        """空数组 content。"""
        messages = [{"role": "user", "content": []}]
        result = _ensure_string_content(messages)
        assert result[0]["content"] == ""


class TestTelemetryStripping:
    """验证 router.prepare_request 早期 telemetry header 剥离。"""

    async def test_billing_header_stripped_from_system(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": (
                    "x-anthropic-billing-header: cc_version=2.1.42.abc;"
                    " cc_entrypoint=claude-code; cch=0;\n"
                    "You are Claude Code."
                )},
                {"role": "user", "content": "hi"},
            ],
        }
        p = await router.prepare_request(body)
        sys_content = p["messages"][0]["content"]
        assert "x-anthropic-billing-header" not in sys_content
        assert "You are Claude Code." in sys_content

    async def test_first_user_message_also_stripped(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": (
                    "x-anthropic-foo: telemetry\nactual question"
                )},
            ],
        }
        p = await router.prepare_request(body)
        # 找到第一条 user 消息（pipeline 可能在前面追加 system）
        first_user = next(m for m in p["messages"] if m.get("role") == "user")
        assert "x-anthropic-foo" not in first_user["content"]
        assert "actual question" in first_user["content"]

    async def test_disabled_by_config(self, cfg: ProxyConfig):
        """关闭开关后 header 透传。"""
        cfg.optimization.strip_client_telemetry = False
        local_router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "x-anthropic-foo: keep\nbase"},
                {"role": "user", "content": "hi"},
            ],
        }
        p = await local_router.prepare_request(body)
        assert "x-anthropic-foo: keep" in p["messages"][0]["content"]

    async def test_no_op_when_no_telemetry(self, router: DeepProxyRouter):
        """普通请求无 header 时，messages 内容除其它 skills 注入外保持稳定。"""
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Plain system."},
                {"role": "user", "content": "hello"},
            ],
        }
        p = await router.prepare_request(body)
        # 不抛、不丢失原文（注：其它 skills 可能在 system 头/尾追加，但保留 'Plain system.'）
        assert "Plain system." in p["messages"][0]["content"]
