"""共享工具函数 —— 通用、模块独立的辅助函数集合。

所有本模块中的函数与 DeepProxy 业务逻辑解耦，可从任何模块导入。
"""

from __future__ import annotations

import asyncio
import hashlib as _hashlib
import json
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 随机区间抽样
# ---------------------------------------------------------------------------


def sample_in_range(lo: float, hi: float) -> float:
    """从 [lo, hi] 均匀抽样并 round 到 0.01。

    lo == hi 是预设里合法的"固定值"形态（如 top_p=[0.95,0.95]、penalties=[0,0]），
    不视为异常；lo > hi 才说明配置错误。
    """
    if lo > hi:
        logger.warning(
            "sample_in_range: lo=%.2f > hi=%.2f（配置非法），退化为定值 %.2f",
            lo, hi, lo,
        )
        return round(lo, 2)
    if lo == hi:
        return round(lo, 2)
    return round(random.uniform(lo, hi), 2)


# ---------------------------------------------------------------------------
# 指数退避重试
# ---------------------------------------------------------------------------


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    backoff_base: float,
    is_retryable: Callable[[Exception], bool],
    label: str = "",
) -> T:
    """通用指数退避重试。第 i 次重试等待 base*(2**i) ± 25% 抖动。

    Args:
        fn: 待重试的异步函数（零参数，闭包封装上下文）。
        max_retries: 最大重试次数（第 1 次重试发生在首次失败后）。
        backoff_base: 退避基数（秒）。
        is_retryable: 决定某异常是否应触发重试。
        label: 日志标签，用于区分调用来源。
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            if attempt >= max_retries or not is_retryable(e):
                raise
            delay = backoff_base * (2**attempt)
            delay *= 1.0 + random.uniform(-0.25, 0.25)
            logger.warning(
                "[%s] 第 %d 次重试，错误: %s，等待 %.2fs",
                label, attempt + 1, e, delay,
            )
            await asyncio.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# SSE 事件格式化
# ---------------------------------------------------------------------------


def format_sse_event(event_name: str, payload: Dict[str, Any]) -> str:
    """格式化 Anthropic / OpenAI 风格的 SSE 事件字符串。

    输出格式：event: {event_name}\\ndata: {json}\\n\\n
    """
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# tool_calls 增量累加
# ---------------------------------------------------------------------------


def merge_tool_call_deltas(
    existing: List[Dict[str, Any]],
    deltas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """OpenAI 流式 tool_calls 按 index 增量累加。

    流式场景中 tool_calls 以 delta 形式逐 chunk 发出，每个 delta 携带
    index 字段标识属于第几个 tool_call。本函数按 index 合并增量。
    """
    by_idx: Dict[int, Dict[str, Any]] = {
        tc.get("index", i): tc for i, tc in enumerate(existing)
    }
    for d in deltas:
        idx = d.get("index", 0)
        cur = by_idx.setdefault(
            idx,
            {"index": idx, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if d.get("id"):
            cur["id"] = d["id"]
        if d.get("type"):
            cur["type"] = d["type"]
        fn = d.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] = fn["name"]
        if isinstance(fn.get("arguments"), str):
            cur["function"]["arguments"] = (
                (cur["function"].get("arguments") or "") + fn["arguments"]
            )
    return [by_idx[k] for k in sorted(by_idx.keys())]


# ---------------------------------------------------------------------------
# URL / 路径工具
# ---------------------------------------------------------------------------


def strip_api_version(base: str) -> str:
    """去掉 URL 路径中的 `/v1` / `/beta` 后缀。"""
    base = base.rstrip("/")
    for suffix in ("/v1", "/beta"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


# ---------------------------------------------------------------------------
# 消息列表操作（system 消息查找与修改）
# ---------------------------------------------------------------------------


def find_system_message(messages: List[Dict[str, Any]]) -> tuple:
    """返回 (首条 system 的 index, content 文本, 是否可压缩)。

    - 无 system 消息 → (None, "", True)
    - content 是字符串 → (i, str, True)
    - content 是 list（多模态）或其他 → (i, "", False)
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return i, content, True
        return i, "", False
    return None, "", True


def append_to_system_message(
    messages: List[Dict[str, object]],
    text: str,
    *,
    dedup: bool = False,
) -> None:
    """把 text 追加到首条 system 消息末尾。

    行为：
    1. 已有 system 且 content 是字符串 → 末尾追加（双换行分隔）
    2. 已有 system 且 dedup=True 且尾部已含 text → 跳过不追加（幂等）
    3. 已有 system 但 content 是非字符串（多模态 list 等）→ 在其前插入新 system
    4. 无 system → 顶部插入新 system

    Args:
        messages: 消息列表（原地修改）。
        text: 要追加的文本。
        dedup: 是否检查 text 是否已出现在 content 中（幂等追加）。
    """
    if not text:
        return
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if dedup and text in content:
                return
            sep = "\n\n" if content else ""
            msg["content"] = f"{content}{sep}{text}"
        else:
            messages.insert(messages.index(msg), {"role": "system", "content": text})
        return
    messages.insert(0, {"role": "system", "content": text})


def prepend_to_system_message(
    messages: List[Dict[str, object]],
    text: str,
) -> None:
    """把 text 插入到首条 system 消息内容的最前面。

    行为：
    1. 已有 system 且 content 是字符串 → 最前面拼接（双换行分隔）
    2. 已有 system 但 content 是非字符串（多模态 list 等）→ 在其前插入新 system
    3. 无 system → 顶部插入新 system
    """
    if not text:
        return
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            sep = "\n\n" if content else ""
            msg["content"] = f"{text}{sep}{content}"
        else:
            messages.insert(messages.index(msg), {"role": "system", "content": text})
        return
    messages.insert(0, {"role": "system", "content": text})


# ---------------------------------------------------------------------------
# 内容提取
# ---------------------------------------------------------------------------


def get_text_from_content(content: Any) -> str:
    """从消息 content 字段提取纯文本字符串。

    OpenAI content 字段可以是：
    - 纯字符串 → 原样返回
    - list[dict]（多模态块数组）→ 提取 type=="text" 的块，换行拼接
    - 其他 → ""

    适用于 user / assistant / system 消息的 content 字段。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# hashlib 工具
# ---------------------------------------------------------------------------


def hash_str(text: str, *, prefix: str = "", algo: str = "sha256") -> str:
    """对字符串取哈希（默认 SHA-256 hexdigest）。"""
    h = _hashlib.new(algo)
    if prefix:
        h.update(prefix.encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def hash_payload(payload: dict, *, prefix: str = "", algo: str = "sha256") -> str:
    """对可 JSON 序列化的 payload 取哈希（默认 SHA-256）。

    JSON 序列化时按 key 排序以确保稳定。
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = _hashlib.new(algo)
    if prefix:
        h.update(prefix.encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 流式协议常量
# ---------------------------------------------------------------------------

SSE_DONE = "data: [DONE]\n\n"
"""OpenAI SSE 协议流结束标记。"""
