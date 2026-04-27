"""带磁盘缓存的 system prompt LLM 压缩器（非阻塞）。

工作流：
1. 命中内存缓存 → 立即返回压缩版（fast path）
2. 未命中 → 立即返回原文（用户请求 0 阻塞）+ 后台异步压缩；下次请求即可命中缓存
3. 同一 key 并发到达 → 仅首个调度后台任务，其余请求正常通过原文

缓存版本号 (CACHE_VERSION) 控制：升级压缩 prompt 或模型时改 +1，旧条目自动失效。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 缓存版本：变更压缩 prompt / 目标模型 / 输出协议时 +1，强制重压缩
_CACHE_VERSION = 1

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```$", re.DOTALL)


def _strip_wrapping(text: str) -> str:
    """剥掉模型偶尔加的代码栅栏 / 前后引号。"""
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1].strip()
    return text

# 压缩 meta-prompt（中文）
# 设计参考：
#   - LLMLingua (Microsoft, 2023)：按 token 自信息排序删冗，保留指令 token > 示例 token
#   - Selective Context (Li et al., 2023)：低自信息 token 优先删
#   - Agent 实践经验：列举常见压缩陷阱（弱化情态、合并不同情形、删除否定 → 改变语义）
_COMPRESSION_INSTRUCTION = """\
你是一个 system prompt 压缩器。请把输入的提示词改写为更短的同义版本，使下游 LLM 在执行改写版时的行为与执行原版时完全一致。

# 目标（按优先级从高到低）

1. 完整保留每一条规则、豁免条款、命名实体、技术术语、格式与输出规范、数值阈值，以及任何作为反模式 / 正例出现的逐字短语。
2. 完整保留每条指令的情态强度。包括英文的 MUST、MUST NOT、SHOULD、MAY、NEVER、"do not"，以及中文的"必须 / 应当 / 宜 / 可以 / 不要 / 严禁"等。否定词承载关键语义，每一处否定都必须保留，不得删除或翻转。
3. 将整体长度压缩到原文 token 数的 40% – 60%。
4. 改写结果整体使用简体中文。原文为英文的部分请翻译为中文；原文已是中文的部分直接精简，不要回译。

# 行文风格（重要）

下游 LLM 会把 system prompt 的语言风格当作生成内容的基线。电报式 prompt（堆砌名词短语、拆成关键词列表、删到只剩骨架）会让模型在创意写作和角色扮演等场景里词汇贫乏、句式单调，因此必须避免。

- 写成完整、自然的中文句子，保留必要的连接词、过渡词和虚词，让规则读起来流畅。
- 简洁应当通过"删除冗余信息"实现，而不是通过"删掉虚词"实现；宁可句子稍长，也要读起来像人话。
- 列表项请写成完整短句，而非孤立的关键词或名词堆叠。

# 必须原样照抄、不翻译也不改写的内容

- 加引号的字符串、代码块、JSON 字段名、XML / HTML 标签名（如 <thinking>、<output>），以及输出格式标记。
- 命名实体、产品名、人名、术语缩写（如 "RP"、"e.g."、"i.e."）。
- 反模式或正例中的逐字短语，这些短语本身就是规则的一部分（例如 'Facts over feelings'、'due to time constraints'）。
- 数值阈值、版本号、日期、URL、DOI。
- 原文中已经是中文的术语、引文和命名条目。

# 可用的压缩手段

- 优先删除真正冗余的部分：重复说明、寒暄客套、问候语、解释规则动机的旁注、举例之前的铺垫。
- 把若干条同主语的祈使句合并为一句，但保留连词使其自然。例如把"不要 X。不要 Y。不要 Z。"改为"不要 X、Y 或 Z"。
- 在含义完全等价时，把长短语替换为业内通用的缩写。
- 允许中英混排：需保留原文的片段保持英文，其余部分改写为中文。

# 严禁的行为

- 不要新增任何指令、削弱原有承诺，或把具体规则改写得更笼统。例如 "Avoid 'Facts over feelings'" 中引号内的英文短语本身就是规则，必须原样保留。
- 不要仅因为关键词相同就合并两条本属不同情形的规则。
- 不要改写引号内的字符串、代码块、命名实体、反模式示例或数值阈值。
- 不要删除或翻转任何否定（"不"、"不要"、"never"、"don't"、"not" 等都不得改动）。
- 不要修改标签名、JSON 字段名或输出格式标记。
- 不要给最终输出包裹代码栅栏、引号或 markdown 标题。
- 不要把句子压成关键词序列、名词堆叠或纯符号串；必须保持自然、可读、流畅的中文。
- 不要写任何前言、解释、注释或后记——你的整个回答就是改写后的提示词本身。

# 输出

仅输出改写后的提示词正文，不附加任何额外内容。"""


class SystemPromptCompressor:
    """带磁盘 + 内存缓存的 LLM-based system prompt 压缩器。"""

    def __init__(
        self,
        *,
        cache_path: Path,
        api_key: str,
        api_base: str,
        model: str = "deepseek/deepseek-v4-flash",
        max_memory: int = 256,
        sampling: Optional[Any] = None,
    ):
        """
        Args:
            sampling: 采样配置实例（PreciseSamplingConfig / CreativeSamplingConfig
                duck-typed，需含 temperature_min/max、top_p_min/max 等字段）。
                None 时退回旧硬编码 temperature=0.1 行为，便于测试隔离。
        """
        self._cache_path = cache_path.resolve()
        self._api_key = api_key
        self._api_base = api_base
        self._model = model
        self._max_memory = max_memory
        self._sampling = sampling
        self._mem: "OrderedDict[str, str]" = OrderedDict()
        # 非阻塞机制：缓存 miss 时后台压缩，主请求路径直接返回原文
        self._inflight: set = set()                 # 正在跑的 cache key（防重复任务）
        self._tasks: set = set()                    # 强引用持有 background tasks
        self._load_disk()
        logger.info(
            "SystemPromptCompressor 已初始化 (model=%s, cache=%s, 已载入 %d 条)",
            self._model, self._cache_path, len(self._mem),
        )

    # ------------------------------------------------------------------ I/O

    def _load_disk(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            if data.get("version") != _CACHE_VERSION:
                logger.info(
                    "压缩缓存版本不匹配（磁盘=%s, 当前=%s），忽略旧缓存",
                    data.get("version"), _CACHE_VERSION,
                )
                return
            entries = data.get("entries", {})
            if isinstance(entries, dict):
                for k, v in entries.items():
                    if isinstance(k, str) and isinstance(v, str):
                        self._mem[k] = v
                logger.info("加载了 %d 条压缩缓存", len(self._mem))
        except Exception as e:
            logger.warning("读取压缩缓存失败 (%s)，忽略: %s", self._cache_path, e)

    def _persist(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            payload = {"version": _CACHE_VERSION, "entries": dict(self._mem)}
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._cache_path)
        except Exception as e:
            logger.warning("写入压缩缓存失败 (%s): %s", self._cache_path, e)

    # --------------------------------------------------------------- Lookup

    @staticmethod
    def _key(text: str, model: str) -> str:
        h = hashlib.sha256()
        h.update(f"v{_CACHE_VERSION}|{model}|".encode("utf-8"))
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    async def compress(self, text: str) -> str:
        """非阻塞压缩入口。

        - 命中内存缓存 → 立即返回压缩版（fast path）
        - 未命中 → 立即返回原文（fail-fast，0 阻塞）+ 在后台异步触发压缩；
          压缩成功后写入缓存，下一次相同请求即可命中
        - 同一 key 并发到达 → 仅首个触发后台任务，其余请求正常通过

        永远不阻塞主请求路径：用户绝不会因为压缩等待 LLM 调用。
        """
        if not text or not text.strip():
            return text
        key = self._key(text, self._model)
        key_short = key[:12]

        # 1. 内存命中：返回压缩版
        if key in self._mem:
            self._mem.move_to_end(key)
            logger.debug("compress: 命中缓存 key=%s, %d 字符", key_short, len(text))
            return self._mem[key]

        # 2. 未命中：调度后台压缩任务（每 key 至多一个 in-flight），主路径返回原文
        if key not in self._inflight:
            self._inflight.add(key)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 不在 event loop 中（极少发生，例如同步上下文）—— 跳过后台压缩
                self._inflight.discard(key)
                logger.debug("compress: 无运行中的 event loop，跳过后台压缩 key=%s", key_short)
                return text
            task = loop.create_task(self._background_compress(text, key))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            logger.info(
                "compress: 缓存未命中 key=%s (%d 字符)，已调度后台压缩；本次请求传输原文",
                key_short, len(text),
            )
        else:
            logger.debug(
                "compress: 缓存未命中 key=%s 但后台任务进行中，本次直接返回原文",
                key_short,
            )

        return text

    async def _background_compress(self, text: str, key: str) -> None:
        """后台执行实际压缩 + 写缓存。所有失败容错开放 + 释放 inflight 槽。"""
        key_short = key[:12]
        try:
            try:
                compressed = await self._call_llm(text)
            except Exception as e:
                logger.warning(
                    "compress (后台): LLM 调用失败 key=%s, %s （不写缓存，下次请求再试）",
                    key_short, e,
                )
                return
            if not compressed or not compressed.strip():
                logger.warning(
                    "compress (后台): LLM 返回空内容 key=%s （不写缓存）", key_short,
                )
                return
            # 写缓存（内存 + 磁盘）
            self._mem[key] = compressed
            while len(self._mem) > self._max_memory:
                self._mem.popitem(last=False)
            self._persist()
            logger.info(
                "compress (后台): 完成 key=%s, %d → %d 字符 (-%d%%)，已写入 %s",
                key_short, len(text), len(compressed),
                int(100 * (1 - len(compressed) / max(1, len(text)))),
                self._cache_path,
            )
        finally:
            self._inflight.discard(key)

    async def wait_inflight(self) -> None:
        """等所有后台压缩任务完成（用于测试 / 关闭时优雅 drain）。"""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ---------------------------------------------------------------- LLM

    async def _call_llm(self, text: str) -> str:
        """直连 LiteLLM 调一次压缩；不通过 router（避免递归注入 skills）。

        max_tokens 容量计算：
          - DeepSeek 服务端**有时会忽略 `thinking={"type":"disabled"}`** 走 enabled 路径，
            把 token 配额耗在 reasoning_content 上、content 被截断为空。
          - 原文 token 估计上限 ≈ 字符数；服务端 reasoning 可能 1-2× 原文；
            实际压缩输出 ≈ 0.5× 原文。给 `4× 字符数 + 4096 安全余量`，足以容纳
            disabled 被忽略时的 reasoning + content。
        """
        import litellm

        # 同时主动请求 reasoning_effort=minimal（即使 thinking=disabled 被忽略，
        # 服务端也会把 reasoning 控制到最短），双重防御。
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _COMPRESSION_INSTRUCTION},
                {"role": "user", "content": text},
            ],
            "max_tokens": max(4096, len(text) * 4 + 4096),
            "thinking": {"type": "disabled", "reasoning_effort": "minimal"},
        }
        if self._sampling is not None:
            from . import sample_in_range
            s = self._sampling
            kwargs["temperature"] = sample_in_range(s.temperature_min, s.temperature_max)
            kwargs["top_p"] = sample_in_range(s.top_p_min, s.top_p_max)
            kwargs["presence_penalty"] = sample_in_range(
                s.presence_penalty_min, s.presence_penalty_max,
            )
            kwargs["frequency_penalty"] = sample_in_range(
                s.frequency_penalty_min, s.frequency_penalty_max,
            )
        else:
            kwargs["temperature"] = 0.1
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        logger.debug(
            "compress LLM call: model=%s api_base=%s max_tokens=%d input_chars=%d",
            self._model, self._api_base, kwargs["max_tokens"], len(text),
        )
        resp = await litellm.acompletion(**kwargs)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            logger.warning("compress LLM 返回 choices 为空: %r", resp)
            return ""

        choice0 = choices[0]
        finish_reason = getattr(choice0, "finish_reason", None)
        msg = getattr(choice0, "message", None)
        content = getattr(msg, "content", None) if msg else None
        # 探测 reasoning_content 仅用于诊断（不能把 reasoning 当 output：CoT 内部
        # 思考内容不是改写后的 prompt 本身）
        reasoning = (
            getattr(msg, "reasoning_content", None) if msg else None
        ) or (getattr(msg, "reasoning", None) if msg else None)
        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "compress LLM 返回空 content: finish_reason=%s, reasoning_len=%s, "
                "completion_tokens=%s, max_tokens=%d, input_chars=%d. "
                "可能原因：thinking=disabled 被服务端忽略 + reasoning 吃满 max_tokens。",
                finish_reason,
                len(reasoning) if isinstance(reasoning, str) else None,
                completion_tokens,
                kwargs["max_tokens"],
                len(text),
            )
            return ""

        if finish_reason == "length":
            # 即使有内容，也警告：被截断的压缩结果可能不完整、不可用
            logger.warning(
                "compress LLM 输出被 max_tokens 截断 (finish_reason=length, "
                "content_len=%d, max_tokens=%d, input_chars=%d)；该结果不写缓存。",
                len(content), kwargs["max_tokens"], len(text),
            )
            return ""

        out = _strip_wrapping(content.strip())
        logger.debug("compress LLM done: %d -> %d chars (finish=%s)",
                     len(text), len(out), finish_reason)
        return out
