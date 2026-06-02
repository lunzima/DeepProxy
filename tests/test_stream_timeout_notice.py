"""iter_chat_chunks 超时优雅通知（根因修复）路由层集成测试。

根因：stream_one_turn / 普通流式路径超时时静默返回空轮，客户端（Claude Code）收到
一次"空且正常结束"的 turn，于是静默停止推理且无错误码。修复后超时改为 yield 优雅
通知（content + finish_reason=stop），让主 agent 知道上游超时、可重试，而非报错或静默。
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.providers import Provider
from deep_proxy.router import DeepProxyRouter


def _notice_present(frames: list[dict]) -> bool:
    return any(
        "[DeepProxy]" in (f.get("choices", [{}])[0].get("delta", {}).get("content") or "")
        for f in frames
    )


def _clean_finish_present(frames: list[dict]) -> bool:
    return any(
        f.get("choices", [{}])[0].get("finish_reason") == "stop" for f in frames
    )


def _no_error_frame(frames: list[dict]) -> bool:
    return not any(
        isinstance(f.get("error"), dict) and not f.get("choices") for f in frames
    )


async def test_plain_path_first_chunk_timeout_emits_graceful_notice(router):
    """cc 未激活的普通流式路径：上游迟迟不给首 chunk → 优雅通知而非静默空轮，
    且不报错（无 error frame）、不提交升格记账。"""
    router.config.streaming.first_chunk_timeout_seconds = 0.2
    router.config.streaming.heartbeat_seconds = 0.1

    async def hang_iter(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(1.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=hang_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=None)]

    assert _notice_present(out)
    assert _clean_finish_present(out)
    assert _no_error_frame(out)
    assert committed["hit"] is False


async def test_plain_path_mid_stream_timeout_emits_notice(router):
    """普通路径：已流出部分内容后相邻 chunk 间断流超 idle_timeout（区别于首 chunk
    超时）→ 先透传已有内容,再注入 mid_stream 通知 + clean finish,不提交升格记账。
    这是唯一端到端触发 idle_timeout_seconds（而非 first_chunk_timeout）接线的路径。"""
    router.config.streaming.first_chunk_timeout_seconds = 5
    router.config.streaming.idle_timeout_seconds = 0.2
    router.config.streaming.heartbeat_seconds = 0.1

    async def slow_iter(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "部分"},
                            "finish_reason": None}]}
        await asyncio.sleep(1.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=slow_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=None)]

    # 已流出的部分内容被透传
    assert any(d.get("choices", [{}])[0].get("delta", {}).get("content") == "部分"
               for d in out)
    # mid_stream 文案（区别于 first_chunk）：含"输出过程中"
    assert any("输出过程中" in d.get("choices", [{}])[0].get("delta", {}).get("content", "")
               for d in out)
    assert _clean_finish_present(out)
    assert _no_error_frame(out)
    assert committed["hit"] is False


async def test_plain_path_normal_stream_unaffected(router):
    """正常流式（首 chunk 及时到达 + 自然 finish）原样透传，不注入通知。"""
    async def ok_iter(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "答案"},
                            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    with patch("deep_proxy.router.iter_litellm_chunks", new=ok_iter):
        out = [f async for f in router.iter_chat_chunks(body, provider=None)]

    assert any(f.get("choices", [{}])[0].get("delta", {}).get("content") == "答案"
               for f in out)
    assert not _notice_present(out)


def _cc_router() -> tuple[DeepProxyRouter, Provider]:
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    cfg.cross_consult = CrossConsultConfig(
        enabled=True, pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    cfg.cross_consult.first_chunk_timeout_seconds = 0.2
    cfg.cross_consult.stream_heartbeat_seconds = 0.1
    cfg.providers["mimo"] = Provider(
        name="mimo", api_base="https://x", api_key="t", litellm_prefix="openai/",
        flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
    )
    return DeepProxyRouter(cfg), cfg.providers["deepseek"]


async def test_cc_initial_turn_timeout_emits_graceful_notice():
    """cross_consult 激活：初始轮首 chunk 超时 → 优雅通知 + clean finish，不报错、
    不提交升格记账。"""
    router, provider = _cc_router()

    async def hang_iter(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(1.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=hang_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=provider)]

    assert _notice_present(out)
    assert _clean_finish_present(out)
    assert _no_error_frame(out)
    assert committed["hit"] is False
