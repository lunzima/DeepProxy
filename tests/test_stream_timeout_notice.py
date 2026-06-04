"""iter_chat_chunks 超时**重试 + 硬错误**路由层集成测试（plain 路径）+ cc 路径通知。

plain 路径根因修复（见 docs/superpowers/specs/2026-06-04-mid-stream-timeout-retry-design.md）：
旧"注入'请重试' content + clean stop"对 agent 结构上不可能触发重试（clean stop = 成功轮），
已废弃。现：pre-content stall → 代理重发（受 max_stream_total_seconds 总预算约束）；
post-content stall / 预算耗尽 → 硬错误帧（{"error": {...}} 透传给客户端使 SDK 抛错）。

cc 路径（cross_consult 激活）已同等迁移：初始轮 + 重发轮经 stream_turn_with_retry
做 pre-content 重试 + 硬错误（见 2026-06-04-cross-consult-retry-design.md）。
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.providers import Provider
from deep_proxy.router import DeepProxyRouter
from deep_proxy.utils import is_error_frame


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
    return not any(is_error_frame(f) for f in frames)


async def test_plain_path_first_chunk_timeout_retries_then_hard_errors(router):
    """plain 路径：上游迟迟不给首 chunk（pre-content）→ 代理在总预算内反复重发；预算
    耗尽 → 硬错误帧透传给客户端（使 SDK 抛错）。不注入旧'请重试'通知/clean stop，
    不提交升格记账。"""
    router.config.streaming.first_chunk_timeout_seconds = 0.2
    router.config.streaming.heartbeat_seconds = 0.1
    router.config.streaming.max_stream_total_seconds = 1   # 1s 总预算 → 快速耗尽
    calls = {"n": 0}

    async def hang_iter(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        await asyncio.sleep(5.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=hang_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=None)]

    assert calls["n"] >= 2                       # 预算内重发了至少一次
    assert not _notice_present(out)              # 无旧'请重试'通知
    assert not _clean_finish_present(out)        # 无误导性 clean stop
    assert any(is_error_frame(f) for f in out)   # 硬错误帧透传给客户端
    assert committed["hit"] is False             # 不提交升格记账


async def test_plain_path_post_content_stall_hard_errors(router):
    """plain 路径：已流出可见 content 后断流超 idle_timeout（post-content，不可续传）→
    立即硬错误帧，**不重发**；已流出内容仍透传；不注入旧通知/clean stop；不提交升格记账。"""
    router.config.streaming.first_chunk_timeout_seconds = 5
    router.config.streaming.idle_timeout_seconds = 0.2
    router.config.streaming.reasoning_idle_timeout_seconds = 0.2
    router.config.streaming.heartbeat_seconds = 0.1
    calls = {"n": 0}

    async def slow_iter(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        yield {"choices": [{"index": 0, "delta": {"content": "部分"},
                            "finish_reason": None}]}
        await asyncio.sleep(5.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=slow_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=None)]

    assert calls["n"] == 1                        # committed 后不重发
    # 已流出的部分内容被透传
    assert any(d.get("choices", [{}])[0].get("delta", {}).get("content") == "部分"
               for d in out)
    assert any(is_error_frame(f) for f in out)    # 硬错误帧
    assert not _notice_present(out)
    assert not _clean_finish_present(out)
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


async def test_cc_initial_turn_timeout_hard_errors():
    """cross_consult 激活：初始轮首 chunk 持续超时、总预算耗尽 → 硬错误帧（透传给
    客户端），不再注入已废弃的优雅通知 / clean stop，不提交升格记账。"""
    router, provider = _cc_router()
    router.config.streaming.max_stream_total_seconds = 1   # 1s 总预算 → 快速耗尽

    async def hang_iter(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(5.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"},
                            "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=hang_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=provider)]

    assert not _notice_present(out)
    assert not _clean_finish_present(out)
    assert not _no_error_frame(out)        # 硬错误帧存在
    assert committed["hit"] is False
