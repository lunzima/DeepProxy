"""Flash→Pro Router（Layer 0）—— 请求前升格决策引擎。

设计对标：RouteLLM（lm-sys/RouteLLM）的 abstract Router + threshold-based routing。

与 RouteLLM 的关键差异：
  - 使用完整对话上下文（messages 数组）而非仅最后一条 prompt
  - 输出为二元决策（upgrade / keep），而非 model name string
  - BERT 为二分类（need_pro / adequate_flash），而非三分类（weak/tie/strong）
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .flash_upgrade import _flatten_messages, compute_complexity_score

logger = logging.getLogger(__name__)

# ======================================================================
# 工具函数
# ======================================================================


def _last_user_content(messages: List[Dict[str, Any]]) -> str:
    """提取最后一条 user 消息的纯文本内容。

    Coding Agent 常将系统上下文（QWEN.md / memory）注入为前几条 user 消息，
    仅取最后一条可隔离真正的用户问题。
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = [
                b.get("text", "")
                for b in c
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            return "\n".join(parts)
    return ""


# ======================================================================
# 抽象基类
# ======================================================================


class UpgradeRouter(ABC):
    """Flash→Pro 路由决策器。

    子类需实现 `should_upgrade(messages) -> float`，返回到 0.0–1.0 的"建议升格"分数。
    外部通过 compare(threshold) 做最终决策：score >= threshold → Pro。
    """

    @abstractmethod
    def score(self, messages: List[Dict[str, Any]], **kwargs) -> float:
        """返回到 0.0–1.0 的建议升格分数。

        Args:
            messages: 完整对话消息列表
            **kwargs: 子类特定参数（如 body 含 extra_body sentinel）

        Returns:
            浮点数，0.0 = 绝对不升格，1.0 = 绝对升格
        """
        ...

    def should_upgrade(
        self,
        messages: List[Dict[str, Any]],
        threshold: float,
        **kwargs,
    ) -> bool:
        """基于 threshold 做出最终路由决策。

        Args:
            threshold: 升格阈值，0.0（全升）~ 1.0（几乎不升）
                与 RouteLLM 语义一致：score >= threshold → 强模型
        """
        return self.score(messages, **kwargs) >= threshold


# ======================================================================
# 规则路由器（永不离线，无需模型）
# ======================================================================


class RuleUpgradeRouter(UpgradeRouter):
    """基于启发式规则的 Router，永不离线。

    将 compute_complexity_score（0–10）缩放到 0.0–1.0，
    同时注入 sentinel/extra_body 等强制升格信号的检测。
    """

    def __init__(self, base_scale: float = 0.1):
        """
        Args:
            base_scale: 将 0–10 的启发式分数映射到 0–1 的缩放因子。
                       默认 0.1 意味着 10 分 → 1.0，6 分 → 0.6，依次类推。
        """
        self._scale = base_scale

    def score(self, messages: List[Dict[str, Any]], **kwargs) -> float:
        # 强制升格信号 → 直接 1.0
        from .flash_upgrade import has_upgrade_sentinel
        if has_upgrade_sentinel(messages):
            return 1.0

        from .flash_upgrade import extra_body_requests_upgrade
        if kwargs.get("body") and extra_body_requests_upgrade(kwargs["body"]):
            return 1.0

        heuristic = compute_complexity_score(messages)
        return round(min(heuristic * self._scale, 1.0), 4)


# ======================================================================
# BERT 路由器（需要 torch + transformers）
# ======================================================================


class BertUpgradeRouter(UpgradeRouter):
    """轻量 BERT 二分类器。

    使用中文 RoBERTa-small + LoRA 做 flash→pro 升格决策（二分类：complex vs simple）。
    CPU 推理约 40-70ms（512 tokens），CUDA 约 10-20ms。

    torch / transformers 为可选项——导入失败或模型加载失败时自动降级到 RuleUpgradeRouter。
    降级链：BertUpgradeRouter → RuleUpgradeRouter（始终可用）。
    """

    def __init__(
        self,
        checkpoint_path: str,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_path: HuggingFace model ID 或本地路径。
                相对路径自动按 CWD 解析；路径不存在时回退到 HF model ID。
            max_length: tokenizer 最大长度
            device: 推理设备（None 自动检测）
        """
        # 相对路径 → 绝对（对 CWD 解析）；HF model ID 原样保留
        self._checkpoint = self._resolve_path(checkpoint_path)
        self._max_length = max_length
        self._device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # 即时加载——首次构造即载入模型，避免首请求冷启动延时
        self._loaded = self._try_load()

    @staticmethod
    def _resolve_path(path: str) -> str:
        """将可能的本地路径解析为绝对路径；非本地路径原样返回。

        规则：
          - 绝对路径 → 原样
          - 本地目录（对 CWD 解析后存在）→ 绝对路径
          - 其余 → 原样返回（可能是 HF model ID 或不存在但合法的路径）
        """
        import os
        if os.path.isabs(path):
            return path
        cwd_candidate = os.path.join(os.getcwd(), path)
        if os.path.isdir(cwd_candidate):
            return cwd_candidate
        # 非本地目录 → 原样（HF model ID 或尚不存在的路径）
        return path

    def score(self, messages: List[Dict[str, Any]], **kwargs) -> float:
        if not self._loaded:
            logger.warning(
                "BERT 模型未加载，降级到规则路由器评分",
            )
            return RuleUpgradeRouter().score(messages, **kwargs)

        # 仅用最后一条 user 消息评分，避免 Coding Agent 注入的巨量
        # 系统上下文（QWEN.md / memory 等）被误判为复杂请求。
        last_user = _last_user_content(messages)
        text = last_user if last_user else _flatten_messages(messages, user_only=True)
        try:
            import torch
        except ImportError:
            logger.warning("torch 不可用，降级到规则路由器")
            return RuleUpgradeRouter().score(messages, **kwargs)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # shape (1, 2): [logit_not_upgrade, logit_upgrade]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            upgrade_prob = probs[0, 1].item()

        # 诊断日志 → logs/deepproxy.log（控制台不刷屏）
        token_count = inputs["input_ids"].shape[1]
        preview = text[:300].replace("\n", "\\n")
        logger.debug(
            "BERT input: chars=%d, tokens=%d, preview=%r, score=%.3f",
            len(text), token_count, preview, upgrade_prob,
        )
        return upgrade_prob

    def _try_load(self) -> bool:
        """加载模型和 tokenizer。返回是否成功。"""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError:
            logger.warning("transformers 未安装，无法加载 BERT 路由器")
            return False

        try:
            logger.info("正在加载 BERT 路由器: %s", self._checkpoint)
            self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._checkpoint,
                num_labels=2,
            )
            self._model.eval()

            if self._device is None:
                try:
                    import torch
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self._device = "cpu"

            if self._device != "cpu":
                self._model.to(self._device)

            logger.info(
                "BERT 路由器已加载: %s (device=%s)",
                self._checkpoint, self._device,
            )
            return True
        except Exception as e:
            logger.error("BERT 路由器加载失败: %s", e)
            self._model = None
            self._tokenizer = None
            return False


# ======================================================================
# 路由选择工厂
# ======================================================================

_ROUTER_REGISTRY: Dict[str, type[UpgradeRouter]] = {
    "rule": RuleUpgradeRouter,
    "bert": BertUpgradeRouter,
}


def create_router(
    kind: str,
    **kwargs,
) -> UpgradeRouter:
    """创建路由器实例。

    Args:
        kind: 路由器类型
            "rule" — 启发式规则（始终可用，无模型）
            "bert" — BERT 二分类（需 torch+transformers）
        **kwargs: 传递给路由器构造函数的参数

    Returns:
        路由器实例。若指定类型加载失败，静默降级到 "rule"。
    """
    cls = _ROUTER_REGISTRY.get(kind)
    if cls is None:
        logger.warning("未知路由器类型 %r，降级到 rule", kind)
        cls = RuleUpgradeRouter

    try:
        return cls(**kwargs)
    except Exception as e:
        logger.warning("路由器 %s 初始化失败 (%s)，降级到 rule", kind, e)
        return RuleUpgradeRouter()
