"""标签触发的整轮 provider 重定向解析。

工作流程：
1. 扫描 body["messages"] 中所有 user 消息，按 cc_config.compiled_redirect_pattern() 匹配
2. 命中 → 设置 RedirectTracker(persist_turns+1)、剥离消息中所有标签匹配、返回目标 provider
3. 未命中 → 询问 RedirectTracker 是否在 persist 窗口内，若是返回目标 provider
4. 都未命中 → 返回 None（不重定向）
5. pairs 中查不到对偶或目标 provider 不存在 → log warning，fail-open（返回 None）

调用方在 main.py::chat_completions 中接收返回值，非 None 时覆盖 provider 后再
调 prepare_request；下游 prepare_request 的所有 provider-aware 步骤（model 名规范化、
thinking 注入、reasoning_effort、flash_upgrade、cross_consult 工具注入）都按
重定向后的 provider 走。
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .redirect_tracker import RedirectTracker

if TYPE_CHECKING:
    from ..config import ProxyConfig
    from ..providers import Provider

logger = logging.getLogger(__name__)


def _strip_one_message(msg: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    """就地剥离单条消息中的所有标签匹配。返回是否命中。"""
    content = msg.get("content")
    if isinstance(content, str):
        if not pattern.search(content):
            return False
        cleaned = pattern.sub("", content).strip()
        msg["content"] = cleaned if cleaned else " "
        return True
    if isinstance(content, list):
        local_hit = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = str(block.get("text", ""))
            if pattern.search(text):
                local_hit = True
                cleaned = pattern.sub("", text).strip()
                block["text"] = cleaned if cleaned else " "
        return local_hit
    return False


def _scan_and_strip(
    messages: list[dict[str, Any]],
    pattern: re.Pattern[str],
) -> bool:
    """扫描 user 消息并就地剥离所有标签匹配。

    返回值仅反映"最后一条 user 消息是否含标签"——这是"本轮是否新触发重定向"
    的判定信号。历史 user 消息中的标签也会被剥离（避免上游模型在 history 中
    看到 meta-instruction 被误导），但不会重新触发重定向计数（持续性由
    RedirectTracker 的 persist_turns 提供，客户端历史回放不应反复重置窗口）。

    多模态 content（list of blocks）按 text 块逐块剥离；纯字符串 content 直接 re.sub。
    剥离后若 content 为空字符串，回填单个空格——避免上游 provider 因空 user 消息报错。
    """
    user_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    if not user_indices:
        return False

    last_idx = user_indices[-1]
    # 先处理"最后一条"——决定本轮是否新触发
    last_hit = _strip_one_message(messages[last_idx], pattern)
    # 历史 user 消息中的标签也清理（不影响触发判定）
    for i in user_indices[:-1]:
        _strip_one_message(messages[i], pattern)
    return last_hit


def resolve_redirect(
    body: dict[str, Any],
    *,
    source_provider: "Provider",
    config: "ProxyConfig",
    tracker: RedirectTracker,
) -> "Provider | None":
    """决策是否需要把请求重定向到异家族 provider。

    Args:
        body: 请求 body（含 messages）。命中时会就地剥离 user 消息中的标签。
        source_provider: 入站 port 绑定的原 provider。
        config: 全局配置，用于查 providers map。
        tracker: 共享 RedirectTracker 实例（由 router 持有）。

    Returns:
        重定向目标 Provider；不重定向时返回 None。
    """
    cc = config.cross_consult
    if not cc.enabled or not cc.redirect_enabled:
        return None

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    # 内部 cross_consult 调用永远不应被重定向（sentinel 防递归）
    if body.get("_deepproxy_cross_consult_internal"):
        return None

    target_name = cc.pair_for(source_provider.name)
    if target_name is None:
        return None  # pairs 未配置或当前 source 无对偶

    target_provider = config.providers.get(target_name)
    if target_provider is None:
        logger.warning(
            "redirect.fail_open source=%s target_name=%s 不在 providers 中，跳过重定向",
            source_provider.name, target_name,
        )
        return None

    pattern = cc.compiled_redirect_pattern()

    # 1) 命中标签 → 重新计数 + 剥离
    if _scan_and_strip(messages, pattern):
        # persist_turns + 1 = 含本次的总轮数；persist_turns=2 → 本次 + 后续 2 轮
        total_turns = cc.redirect_persist_turns + 1
        tracker.set_remaining(
            messages, total_turns,
            source_provider_name=source_provider.name,
        )
        logger.info(
            "redirect.triggered source=%s → target=%s (windows=%d turns)",
            source_provider.name, target_name, total_turns,
        )
        return target_provider

    # 2) 未命中 → 检查是否仍在 persist 窗口（一次调用拿 active + 剩余轮数）
    active, remaining = tracker.consume_turn(
        messages, source_provider_name=source_provider.name,
    )
    if active:
        logger.info(
            "redirect.persist_continue source=%s → target=%s (remaining=%d)",
            source_provider.name, target_name, remaining,
        )
        return target_provider

    return None


__all__ = ["resolve_redirect", "_scan_and_strip"]
