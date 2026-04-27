"""消息转换 skills —— RE2 / CoT Reflection / readurls。

这些技能在 `apply_cheap_optimizations` 的 D 组中按条件激活，
直接改写 messages 内容而非系统提示注入。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .skills_general import _COT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── RE2 ──────────────────────────────────────────────────────────────────

_RE2_MARKER = "\n再读一遍这个问题："


def _apply_re2(messages: List[Dict[str, Any]]) -> None:
    """复制最后一条 user 消息内容（optillm 的 RE2 算法核心，提示词中文化）。"""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return
        # 已经复制过则跳过（idempotent）
        if _RE2_MARKER in content:
            return
        msg["content"] = f"{content}{_RE2_MARKER}{content}"
        return


# ── CoT Reflection ──────────────────────────────────────────────────────

_OUTPUT_TAG_RE = re.compile(r"<output>(.*?)(?:</output>|$)", re.DOTALL)


def _apply_cot_reflection(messages: List[Dict[str, Any]]) -> None:
    """注入 CoT Reflection 引导的 system 提示。

    若已有 system 消息，把 CoT 提示叠加到其前；否则新增一条 system 消息。
    """
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and _COT_SYSTEM_PROMPT not in content:
                msg["content"] = f"{_COT_SYSTEM_PROMPT}\n\n{content}"
            return
    messages.insert(0, {"role": "system", "content": _COT_SYSTEM_PROMPT})


def extract_cot_output(content: str) -> str:
    """从含 `<output>` 标签的模型回复里提取最终答案。

    无标签时原样返回（fail-open，避免吞掉模型未遵循指令时的有效内容）。
    """
    if not content or "<output>" not in content:
        return content
    match = _OUTPUT_TAG_RE.search(content)
    if not match:
        return content
    extracted = match.group(1).strip()
    return extracted or content


# ── readurls ─────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\'\"<>)]+")
_READURLS_MARKER = "[Content from "  # 用于 idempotent 检测
_READURLS_MAX_LEN = 8000  # 每个 URL 最多内联多少字符
_READURLS_TIMEOUT = 5.0  # 单个 URL 抓取超时（秒）
_READURLS_MAX_PER_MSG = 6  # 单条消息最多抓多少个 URL（防滥发链接 → 串行超时累积）
_READURLS_MAX_BYTES = 2 * 1024 * 1024  # 单 URL 响应字节上限（防内存爆炸；2 MiB 足够纯文本）
_READURLS_OK_CT_PREFIXES = (
    "text/html", "text/plain", "application/xhtml", "application/json", "text/xml",
)


async def _apply_readurls(
    messages: List[Dict[str, Any]],
    *,
    client: httpx.AsyncClient | None,
) -> None:
    """对所有 user 消息抓取并内联其中 URL 的正文（optillm/plugins/readurls_plugin.py 同构）。

    健壮性原则（fail-open）：
    - 任何单个 URL 抓取/解析的异常被吞在 `_fetch_url_text` 内（含 CancelledError 透传）
    - 单条 message 处理崩溃不影响后续 messages
    - client 创建/关闭异常不影响整体流程，最坏情况 readurls 整体跳过
    - 同一消息内多 URL 并发抓取，不被慢站点串行阻塞
    """
    own_client = False
    if client is None:
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(_READURLS_TIMEOUT),
                follow_redirects=True,
            )
            own_client = True
        except Exception as e:
            logger.debug("readurls: httpx.AsyncClient 创建失败，跳过整轮 readurls: %s", e)
            return

    try:
        for msg in messages:
            try:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if not isinstance(content, str) or not content:
                    continue
                # 已内联过则跳过（idempotent）
                if _READURLS_MARKER in content:
                    continue
                urls = _URL_RE.findall(content) or []
                if not urls:
                    continue
                # 去重 + 上限：防滥发链接拖垮整请求（每个 URL 最多 _READURLS_TIMEOUT 秒）
                seen: set[str] = set()
                clean_urls: List[str] = []
                for url in urls:
                    cu = url.rstrip(",.;:'\"!?)]}")
                    if not cu or cu in seen:
                        continue
                    seen.add(cu)
                    clean_urls.append(cu)
                    if len(clean_urls) >= _READURLS_MAX_PER_MSG:
                        break

                # 并发抓取（return_exceptions=True：单 URL 异常不影响其它）
                results = await asyncio.gather(
                    *(_fetch_url_text(client, u) for u in clean_urls),
                    return_exceptions=True,
                )

                modified = content
                for cu, res in zip(clean_urls, results):
                    if isinstance(res, BaseException):
                        # asyncio.CancelledError 也属 BaseException 的派生（>=3.8）
                        if isinstance(res, asyncio.CancelledError):
                            raise res
                        logger.debug("readurls: %s 抓取异常被吞: %r", cu, res)
                        continue
                    snippet = res or ""
                    if not snippet:
                        continue
                    try:
                        domain = urlparse(cu).netloc or "url"
                    except Exception:
                        domain = "url"
                    replacement = f"{cu} [Content from {domain}: {snippet}]"
                    modified = modified.replace(cu, replacement, 1)

                if modified != content:
                    msg["content"] = modified
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # 单条 message 处理崩溃 — 跳过该 message，继续后续
                logger.warning("readurls: message 处理异常已跳过: %r", e)
                continue
    finally:
        if own_client:
            try:
                await client.aclose()
            except Exception as e:
                logger.debug("readurls: client.aclose() 异常（已忽略）: %s", e)


async def _fetch_url_text(client: httpx.AsyncClient, url: str) -> str:
    """抓取 URL，返回剥离 HTML 后的纯文本片段；任何失败返回空串。

    多重防御：
    - 网络/超时/连接错误 → 返回 ""
    - 非文本 Content-Type（图片/PDF/二进制）→ 返回 ""
    - 响应体超过 _READURLS_MAX_BYTES → 截断，不读全
    - HTML 解析异常 / get_text 异常 / 文本压缩异常 → 各自捕获，返回空串或安全降级
    - asyncio.CancelledError 透传（不阻挡上层取消）
    """
    # 1. 仅接受 http / https（_URL_RE 已限定，但二次防御）
    try:
        scheme = urlparse(url).scheme.lower()
    except Exception:
        return ""
    if scheme not in ("http", "https"):
        return ""

    # 2. 流式抓取：拿到响应后看 Content-Type 决定是否继续读 body；同时限制总字节数
    raw: bytes = b""
    try:
        async with client.stream(
            "GET", url,
            headers={"user-agent": "deepproxy-readurls/1.0", "accept": "text/html, */*"},
        ) as resp:
            resp.raise_for_status()
            ct = (resp.headers.get("content-type") or "").lower()
            if ct and not any(ct.startswith(p) for p in _READURLS_OK_CT_PREFIXES):
                return ""
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                raw += chunk
                if len(raw) >= _READURLS_MAX_BYTES:
                    break
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("readurls 抓取失败 %s: %s", url, e)
        return ""

    if not raw:
        return ""

    # 3. HTML 解析：lxml 异常 → 退回 html.parser；再失败则返回空串
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(raw, "html.parser")
        except Exception as e:
            logger.debug("readurls HTML 解析失败 %s: %s", url, e)
            return ""

    # 4. 清理脚本/样式（decompose 异常通常源于损坏的 DOM；逐个 try）
    try:
        for tag in soup(["script", "style", "noscript"]):
            try:
                tag.decompose()
            except Exception:
                continue
    except Exception:
        pass

    # 5. 抽文本 + 折叠空白
    try:
        text = soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.debug("readurls get_text 失败 %s: %s", url, e)
        return ""
    try:
        text = re.sub(r"\s+", " ", text).strip()
    except Exception:
        text = text.strip() if isinstance(text, str) else ""

    if not text:
        return ""
    if len(text) > _READURLS_MAX_LEN:
        text = text[:_READURLS_MAX_LEN] + "..."
    return text
