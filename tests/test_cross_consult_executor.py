"""Cross-Consult executor 单测（mock 流式 iter_litellm_chunks）。"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.cross_consult.executor import execute_consult
from deep_proxy.providers import Provider


@pytest.fixture
def cfg_for_executor():
    return ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))


@pytest.fixture
def target_provider():
    return Provider(
        name="mimo",
        api_base="https://x",
        api_key="tp",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
        allowed_extra_params=["reasoning_effort", "thinking"],
    )


def _text_chunks(text: str):
    """生成两个 chunk：role + content delta，模拟最小流式响应。"""
    async def gen(config, body, *, provider=None):
        yield {"choices": [{"index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None}]}
        yield {"choices": [{"index": 0,
                            "delta": {"content": text},
                            "finish_reason": "stop"}]}
    return gen


def _capture_body_iter(captured: list, text: str = "ok"):
    """把上游收到的 body 抓出来，再 yield 一个最小流式响应。"""
    async def gen(config, body, *, provider=None):
        captured.append(body)
        yield {"choices": [{"index": 0,
                            "delta": {"content": text},
                            "finish_reason": "stop"}]}
    return gen


async def test_execute_consult_calls_target_pro_model(cfg_for_executor, target_provider):
    """consult 调用必须指向 target_provider.pro_model，不是 flash_model。"""
    captured: list[dict] = []
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="external answer"),
    ):
        out = await execute_consult(
            question="What is 2+2?",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert out == "external answer"
    assert captured and captured[0]["model"] == "mimo-v2.5-pro"


async def test_execute_consult_sets_recursion_sentinel(cfg_for_executor, target_provider):
    """consult 调用 body 必须带 _deepproxy_cross_consult_internal=True（防递归注入）。"""
    captured: list[dict] = []
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="x"),
    ):
        await execute_consult(
            question="hi",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert captured[0].get("_deepproxy_cross_consult_internal") is True


async def test_execute_consult_no_tools_in_body(cfg_for_executor, target_provider):
    """consult 调用绝不携带 tools 数组（防递归）。"""
    captured: list[dict] = []
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="x"),
    ):
        await execute_consult(
            question="hi",
            context=None,
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    body = captured[0]
    assert "tools" not in body
    assert "tool_choice" not in body


async def test_execute_consult_includes_context_when_given(cfg_for_executor, target_provider):
    """context 不为空时应附加到 user message。"""
    captured: list[dict] = []
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="ok"),
    ):
        await execute_consult(
            question="why?",
            context="some background",
            target_provider=target_provider,
            config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    user_msg = captured[0]["messages"][-1]["content"]
    assert "why?" in user_msg
    assert "some background" in user_msg


async def test_execute_consult_uses_streaming_path(cfg_for_executor, target_provider):
    """关键回归：execute_consult 必须经过 streaming.iter_litellm_chunks，
    不再调用同步的 call_litellm（避免墙钟超时杀掉深度思考）。"""
    chunks_seen = {"calls": 0}

    async def stream_gen(config, body, *, provider=None):
        chunks_seen["calls"] += 1
        yield {"choices": [{"index": 0,
                            "delta": {"content": "ans"},
                            "finish_reason": "stop"}]}

    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=stream_gen,
    ):
        out = await execute_consult(
            question="hi", context=None,
            target_provider=target_provider, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert out == "ans"
    assert chunks_seen["calls"] == 1


async def test_execute_consult_accumulates_reasoning_when_content_empty(
    cfg_for_executor, target_provider,
):
    """模型把全部内容塞进 reasoning_content（MiMo 某些 prompt 下出现）时，兜底取出。"""
    async def stream_gen(config, body, *, provider=None):
        yield {"choices": [{"index": 0,
                            "delta": {"reasoning_content": "deep think"},
                            "finish_reason": "stop"}]}

    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=stream_gen,
    ):
        out = await execute_consult(
            question="hi", context=None,
            target_provider=target_provider, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert out == "deep think"


async def test_execute_consult_max_tokens_is_provider_ceiling(cfg_for_executor):
    """consult 的 max_tokens 用 target_provider 的真实输出上限，而非一个武断的小常量。

    旧实现写死 cc_config.max_output_tokens（4096/16384 之类的魔法数），既无依据又会
    在 reasoning 吃预算时截断答案。真正的约束是 provider 自己的输出上限。
    """
    captured: list[dict] = []
    provider = Provider(
        name="mimo", api_base="https://x", api_key="tp", litellm_prefix="openai/",
        flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
        max_output_tokens=128000,
        reasoning_effort_field="reasoning_effort", reasoning_effort_value="high",
        allowed_extra_params=["reasoning_effort", "thinking"],
    )
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="ok"),
    ):
        await execute_consult(
            question="hi", context=None,
            target_provider=provider, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert captured[0]["max_tokens"] == 128000


async def test_execute_consult_enables_reasoning_top_level(cfg_for_executor):
    """根因回归：consult 调用必须为 reasoning provider 注入 reasoning_effort。

    target 走顶层 reasoning_effort 协议（MiMo 风格）时，consult body 必须带顶层
    reasoning_effort。否则 reasoning 模型静默思考、不流式吐 token，首 chunk 迟迟不
    到 → 误触 first_chunk 超时（cross_consult 超时根因）。
    """
    captured: list[dict] = []
    mimo_top_level = Provider(
        name="mimo",
        api_base="https://x",
        api_key="tp",
        litellm_prefix="openai/",
        flash_model="mimo-v2.5",
        pro_model="mimo-v2.5-pro",
        reasoning_effort_field="reasoning_effort",
        reasoning_effort_value="high",
        allowed_extra_params=["reasoning_effort", "thinking"],
    )
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="ok"),
    ):
        await execute_consult(
            question="hi", context=None,
            target_provider=mimo_top_level, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert captured[0].get("reasoning_effort") == "high"


async def test_execute_consult_enables_reasoning_nested(cfg_for_executor):
    """根因回归：target 走嵌套 thinking.reasoning_effort 协议（DeepSeek 风格）时，
    consult body 必须带 thinking.reasoning_effort + type=enabled。"""
    captured: list[dict] = []
    deepseek_nested = Provider(
        name="deepseek",
        api_base="https://api.deepseek.com",
        api_key="sk",
        litellm_prefix="deepseek/",
        flash_model="deepseek-v4-flash",
        pro_model="deepseek-v4-pro",
        reasoning_effort_field="thinking.reasoning_effort",
        reasoning_effort_value="max",
    )
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=_capture_body_iter(captured, text="ok"),
    ):
        await execute_consult(
            question="hi", context=None,
            target_provider=deepseek_nested, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    thinking = captured[0].get("thinking")
    assert isinstance(thinking, dict)
    assert thinking.get("reasoning_effort") == "max"
    assert thinking.get("type") == "enabled"


async def test_execute_consult_returns_error_string_on_idle_timeout(
    cfg_for_executor, target_provider,
):
    """首 chunk 迟迟不到（连 first_chunk_timeout 都超），返回带前缀的错误字符串，不抛异常。"""
    cfg_for_executor.streaming.first_chunk_timeout_seconds = 0.5
    async def slow_stream(config, body, *, provider=None):
        # 模拟连接成功但 chunk 永不到达——首 chunk 超时路径
        await asyncio.sleep(2.0)
        yield {"choices": [{"index": 0,
                            "delta": {"content": "should not see"},
                            "finish_reason": "stop"}]}

    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=slow_stream,
    ):
        out = await execute_consult(
            question="hi", context=None,
            target_provider=target_provider, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert out.startswith("[DeepProxy cross_consult error]")
    assert "timeout" in out.lower()


async def test_execute_consult_streams_long_thinking_without_timeout(
    cfg_for_executor, target_provider,
):
    """关键回归：模型流式吐了一长串 reasoning（每个 chunk 间隔短），即使总耗时超过
    单个 idle 预算也不应被超时——只要 chunk 持续到达。"""
    cfg_for_executor.streaming.first_chunk_timeout_seconds = 1
    cfg_for_executor.streaming.idle_timeout_seconds = 0.2
    cfg_for_executor.streaming.reasoning_idle_timeout_seconds = 0.5
    async def long_thinking_stream(config, body, *, provider=None):
        for i in range(5):
            await asyncio.sleep(0.1)  # 每 chunk 100ms，5 chunk → ~500ms 总
            yield {"choices": [{"index": 0,
                                "delta": {"reasoning_content": f"step{i} "},
                                "finish_reason": None}]}
        yield {"choices": [{"index": 0,
                            "delta": {"content": "final"},
                            "finish_reason": "stop"}]}

    # idle timeout = 200ms：相邻 chunk 间 100ms < 200ms，应全程通过
    with patch(
        "deep_proxy.cross_consult.streaming.iter_litellm_chunks",
        new=long_thinking_stream,
    ):
        out = await execute_consult(
            question="hi", context=None,
            target_provider=target_provider, config=cfg_for_executor,
            cc_config=CrossConsultConfig(enabled=True),
        )
    assert out == "final"
