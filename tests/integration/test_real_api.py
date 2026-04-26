"""真实 DeepSeek API 集成测试。

默认 SKIP；设置环境变量 `DEEPSEEK_API_KEY` 后自动启用。
跑法：
    DEEPSEEK_API_KEY=sk-xxx python -m pytest tests/integration -v
"""
import os

import pytest

from deep_proxy.config import DeepSeekConfig, ProxyConfig
from deep_proxy.router import DeepProxyRouter

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

pytestmark = pytest.mark.skipif(
    not API_KEY, reason="未设置 DEEPSEEK_API_KEY，跳过真实 API 集成测试"
)


@pytest.fixture
def router_real() -> DeepProxyRouter:
    cfg = ProxyConfig(deepseek=DeepSeekConfig(api_key=API_KEY or "sk-test"))
    return DeepProxyRouter(cfg)


class TestThinkingDisabled:
    """V4 thinking=disabled — 普通对话路径。"""

    async def test_basic_completion(self, router_real: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "回答一个字: 是"}],
            "max_tokens": 5,
        }
        prepared = await router_real.prepare_request(body)
        result = await router_real.chat_completions(prepared)
        assert result["choices"][0]["message"]["content"]


class TestThinkingEnabled:
    """V4 thinking=enabled — 验证 reasoning_content 被保留。"""

    async def test_reasoning_content_present(self, router_real: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "1+1 等于几？"}],
            "max_tokens": 50,
        }
        prepared = await router_real.prepare_request(body)
        result = await router_real.chat_completions(prepared)
        msg = result["choices"][0]["message"]
        assert "reasoning_content" in msg, (
            "V4-Pro thinking=enabled 响应缺 reasoning_content；"
            " 检查 LiteLLM 是否在 model_dump 时剥离了字段。"
        )
        # 兼容字段也应同时存在
        assert msg.get("reasoning") == msg["reasoning_content"]


class TestMultiTurnReasoningPersistence:
    """多轮 + reasoning_content 回传。"""

    async def test_round_trip_with_reasoning(self, router_real: DeepProxyRouter):
        body1 = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "记住数字 7"}],
            "max_tokens": 30,
        }
        prepared = await router_real.prepare_request(body1)
        r1 = await router_real.chat_completions(prepared)
        msg1 = r1["choices"][0]["message"]

        body2 = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "enabled"},
            "messages": [
                {"role": "user", "content": "记住数字 7"},
                # 关键：原样回传 assistant + reasoning_content
                {**msg1},
                {"role": "user", "content": "你刚才记住的是哪个数字？"},
            ],
            "max_tokens": 30,
        }
        prepared2 = await router_real.prepare_request(body2)
        # 不应抛 400
        r2 = await router_real.chat_completions(prepared2)
        assert "7" in r2["choices"][0]["message"]["content"]


class TestStreaming:
    """流式 + reasoning_content delta 透传。"""

    async def test_streaming_chunks(self, router_real: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 20,
            "stream": True,
        }
        prepared = await router_real.prepare_request(body)
        chunks = []
        async for chunk in router_real.chat_completions_stream(prepared):
            chunks.append(chunk)
        assert chunks
        # 至少有一个 chunk 包含 reasoning_content（V4-Pro thinking）
        joined = b"".join(chunks).decode("utf-8", errors="replace")
        assert "reasoning_content" in joined or "reasoning" in joined
