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

from ..utils import find_system_message, prepend_to_system_message
from .skills_general import _COT_SYSTEM_PROMPT, _READURLS_MARKER

logger = logging.getLogger(__name__)

# ── RE2 ──────────────────────────────────────────────────────────────────

_RE2_MARKER = "\n请再阅读一遍上面的内容，然后作答：\n"


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
# 剥离 thinking/reflection 区块及其内部文本 + 标签残余
_COT_BLOCK_RE = re.compile(r"<(thinking|reflection)>.*?</\1>\s*", re.DOTALL)
_COT_TAG_RE = re.compile(r"</?(?:thinking|reflection|output)>", re.IGNORECASE)


def _apply_cot_reflection(messages: List[Dict[str, Any]]) -> None:
    """注入 CoT Reflection 引导的 system 提示。

    使用 prepend_to_system_message 统一插入逻辑。
    """
    _idx, content, _compressible = find_system_message(messages)
    if content and _COT_SYSTEM_PROMPT in content:
        return  # 已存在，跳过
    prepend_to_system_message(messages, _COT_SYSTEM_PROMPT)


def extract_cot_output(content: str) -> str:
    """从含 `<output>` 标签的模型回复里提取最终答案。

    无标签时原样返回（fail-open，避免吞掉模型未遵循指令时的有效内容）。
    `<output>` 标签为空或纯空白时，剥离 thinking/reflection/output 标签
    而非直接返回带标签的原文（避免内部推理泄漏）。
    """
    if not content or "<output>" not in content:
        return content
    match = _OUTPUT_TAG_RE.search(content)
    if not match:
        return content
    extracted = match.group(1).strip()
    if extracted:
        return extracted
    # 空输出：剥离 thinking/reflection 区块，清理标签残余
    stripped = _COT_BLOCK_RE.sub("", content)
    stripped = _COT_TAG_RE.sub("", stripped).strip()
    return stripped or ""


# ── readurls ─────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\'\"<>)]+")
_READURLS_MAX_LEN = 8000  # 每个 URL 最多内联多少字符
_READURLS_TIMEOUT = 5.0  # 单个 URL 抓取超时（秒）
_READURLS_MAX_PER_MSG = 6  # 单条消息最多抓多少个 URL（防滥发链接 → 串行超时累积）
_READURLS_MAX_BYTES = 2 * 1024 * 1024  # 单 URL 响应字节上限（防内存爆炸；2 MiB 足够纯文本）
_READURLS_OK_CT_PREFIXES = (
    "text/html", "text/plain", "application/xhtml", "application/json", "text/xml",
)


def _extract_urls(content: str) -> List[str]:
    """从 content 中提取去重 + 上限截断后的待抓取 URL 列表。

    纯函数，无 I/O；抽出便于单元测试 URL 解析 / 去重 / 上限逻辑。
    """
    raw = _URL_RE.findall(content) or []
    seen: set[str] = set()
    clean: List[str] = []
    for url in raw:
        cu = url.rstrip(",.;:'\"!?)]}")
        if not cu or cu in seen:
            continue
        seen.add(cu)
        clean.append(cu)
        if len(clean) >= _READURLS_MAX_PER_MSG:
            break
    return clean


async def _fetch_all(
    client: httpx.AsyncClient, urls: List[str],
) -> List[Any]:
    """并发抓取所有 URL；per-URL 严格 timeout = _READURLS_TIMEOUT。

    return_exceptions=True：单 URL 异常不阻断其它；上层按异常类型分流
    （CancelledError 透传，其它吞掉）。asyncio.wait_for 防共享 httpx 客户端
    的全局 10s timeout 让慢站点逃过 _READURLS_TIMEOUT 上限。
    """
    return await asyncio.gather(
        *(
            asyncio.wait_for(_fetch_url_text(client, u), timeout=_READURLS_TIMEOUT)
            for u in urls
        ),
        return_exceptions=True,
    )


def _substitute_urls(content: str, urls: List[str], results: List[Any]) -> str:
    """把 (url, fetched_snippet) 对替换进 content；异常/空抓取保持原样。

    纯函数（除了让 CancelledError 透传给调用方处理）。
    """
    modified = content
    for cu, res in zip(urls, results):
        if isinstance(res, BaseException):
            # CancelledError 必须重新抛出让上层 task 取消正常传播
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
        # 负向先行断言：仅在 cu 后面不是 URL 续接字符时替换，避免一个 URL 是另一个的前缀时
        # （https://a.com vs https://a.com/page）短 URL 的替换切进长 URL 中间。
        # 用函数替换而非字符串替换：snippet 里的 \1 / \g<0> 等不会被当作正则回引解释。
        pattern = re.compile(re.escape(cu) + r"(?![\w./?#%=&~+:@!$*,;()\[\]-])")
        modified = pattern.sub(lambda _m: replacement, modified, count=1)
    return modified


async def _process_one_message(
    msg: Dict[str, Any], client: httpx.AsyncClient,
) -> None:
    """对单条 user 消息执行 URL 抓取 + 内联替换。

    仅处理 string content 的 user 消息；非 user / 空 / 已内联过 / 无 URL
    时直接 no-op。原地 mutate msg['content']。
    """
    if msg.get("role") != "user":
        return
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        return
    if _READURLS_MARKER in content:  # 已内联过则跳过（idempotent）
        return
    urls = _extract_urls(content)
    if not urls:
        return

    results = await _fetch_all(client, urls)
    modified = _substitute_urls(content, urls, results)
    if modified != content:
        msg["content"] = modified


async def _apply_readurls(
    messages: List[Dict[str, Any]],
    *,
    client: httpx.AsyncClient | None,
) -> None:
    """对所有 user 消息抓取并内联其中 URL 的正文（optillm/plugins/readurls_plugin.py 同构）。

    顶层只负责 httpx 客户端生命周期 + 跨 messages 的 fail-open 隔离；
    单条 message 的 URL 提取/抓取/替换逻辑见 _process_one_message。

    健壮性原则（fail-open）：
    - 任何单个 URL 抓取/解析的异常被吞在 _fetch_url_text + _substitute_urls 内
    - 单条 message 处理崩溃不影响后续 messages
    - client 创建/关闭异常不影响整体流程，最坏情况 readurls 整体跳过
    - CancelledError 始终透传以让 task 取消正常传播
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
                await _process_one_message(msg, client)
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
