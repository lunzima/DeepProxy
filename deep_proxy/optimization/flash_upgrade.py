"""Flash→Pro 选择性升格机制。

四层架构（全部在 prepare_request 中 upfront 完成，零流式侵入）：
  Layer 0: Router（轻量 BERT classifier，upfront 决策）
  Layer 1: 启发式预检（零成本快速路径，高确信度直接升格）
  Layer 2: Router 执行（改写 body["model"] 为 pro）
  Layer 3: 对话级持久化（UpgradeTracker 按对话指纹保持 Pro N 轮）

设计对标：RouteLLM（lm-sys/RouteLLM）的 proxy-level upfront routing 模式。
与 RouteLLM 的关键区别：
  - 使用完整对话上下文而非仅最后一条 message
  - 集成到 DeepProxy 的 prepare_request 管道，而非独立的 Controller
  - 三层触发（sentinel/启发式/Router），而非单一 threshold 模型
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple


# ======================================================================
# Layer 3：对话级持久化 — UpgradeTracker
# ======================================================================


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    """提取最后一条 user 消息的纯文本内容。"""
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


def _count_user_messages(messages: List[Dict[str, Any]]) -> int:
    """统计 user 消息数量。"""
    return sum(1 for m in messages if m.get("role") == "user")


def _last_user_hash(messages: List[Dict[str, Any]]) -> str:
    """最后一条 user 消息的短哈希（检测重复消息用）。"""
    text = _last_user_text(messages)
    return hashlib.md5(text.encode()).hexdigest()[:8] if text else "empty"


class DailyUpgradeThrottle:
    """防刷屏保护：同一 user 消息连续触发升格 N 次后，强制降级 Flash + 冷却。

    Coding Agent 场景下同一复杂 prompt 可能被重复提交多次，
    每次 BERT/启发式都会打高分升格到 Pro，造成浪费。

    规则：
      - 同一 user hash 连续触发升格 ≥ max_repeats 次 → 强制 Flash
      - 冷却 cooldown_turns 轮（期间新消息不计数，直接走 Flash）
      - 冷却结束后自动恢复，计数器清零
    """

    def __init__(self, max_repeats: int = 5, cooldown_turns: int = 3):
        self._max = max_repeats
        self._cooldown = cooldown_turns
        # hash → (consecutive_upgrade_count, cooldown_remaining)
        self._state: Dict[str, Tuple[int, int]] = {}

    def should_throttle(
        self, messages: List[Dict[str, Any]], did_upgrade: bool
    ) -> bool:
        """检查是否应强制降级。

        Args:
            messages: 当前 messages 数组
            did_upgrade: 上层路由通路（启发式/BERT）是否判定为升格

        Returns:
            True = 强制使用 Flash（降级）；False = 走正常逻辑
        """
        h = _last_user_hash(messages)
        entry = self._state.get(h)

        if entry is None:
            self._state[h] = (1 if did_upgrade else 0, 0)
            return False

        count, cooldown = entry
        if cooldown > 0:
            # 冷却中：强制 Flash，扣一轮
            self._state[h] = (0, cooldown - 1)
            return True

        if did_upgrade:
            count += 1
            if count >= self._max:
                # 冷却 3 轮
                self._state[h] = (0, self._cooldown - 1)
                return True
            self._state[h] = (count, 0)
        else:
            # 没升格 → 序列中断，归零
            self._state[h] = (0, 0)

        return False


def conversation_fingerprint(messages: List[Dict[str, Any]]) -> str:
    """跨轮次稳定的对话标识，不依赖客户端会话 ID。

    仅使用首条 user 内容[:300] 的 md5 —— 从对话第一轮就确定，永不变化。
    单用户场景碰撞概率可忽略；若真正发生（两对话首条完全相同），
    最坏情况是共享升格状态，成本可接受。

    注意：不使用 assistant 内容做 key，因为首轮升格触发时 assistant 尚不存在，
    后续 fingerprint 会改变，导致 UpgradeTracker 找不到对应 key。
    """
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user is None:
        return hashlib.md5(b"empty").hexdigest()

    c = first_user.get("content", "")
    prefix = c[:300] if isinstance(c, str) else str(c)[:300]
    return hashlib.md5(prefix.encode()).hexdigest()


class UpgradeTracker:
    """按对话指纹跟踪 Flash→Pro 持久升格状态。

    每轮新请求（消息数组长度增长）消耗 1 轮剩余额度。
    重试/同轮次请求不消耗。
    并发安全：不同对话的不同指纹天然隔离。

    Examples:
        >>> tracker = UpgradeTracker()
        >>> msgs = [{"role": "user", "content": "写个排序算法"}]
        >>> tracker.set_remaining(msgs, 2)   # 还能用 Pro 2 轮
        >>> tracker.is_upgraded(msgs)         # 同一轮 → True
        True
        >>> msgs.append({"role": "assistant", "content": "..."})
        >>> msgs.append({"role": "user", "content": "优化它"})
        >>> tracker.is_upgraded(msgs)         # 新轮次 → 消耗 1，剩余 1
        True
    """

    def __init__(self, max_size: int = 512):
        self._sessions: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self._max = max_size

    # -- 公开 API --

    def is_upgraded(self, messages: List[Dict[str, Any]]) -> bool:
        """当前对话是否处于升格状态。

        副作用：如果这不是重试（消息数组变长），消耗 1 轮剩余额度。
        """
        fp = conversation_fingerprint(messages)
        entry = self._sessions.get(fp)
        if entry is None:
            return False

        remaining, last_count = entry
        if len(messages) > last_count:
            remaining -= 1
        if remaining <= 0:
            del self._sessions[fp]
            return False
        # 只有当轮次变化时才更新 last_count
        if len(messages) > last_count:
            self._sessions[fp] = (remaining, len(messages))
        return True

    def set_remaining(self, messages: List[Dict[str, Any]], turns: int) -> None:
        """升格触发后记录剩余 Pro 轮次。

        Args:
            turns: 当前请求**之后**还能使用 Pro 的轮次数。
                   例如 turns=2 表示当前请求走 Pro + 后续 2 轮。
        """
        fp = conversation_fingerprint(messages)
        self._sessions[fp] = (turns, len(messages))
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    def remaining(self, messages: List[Dict[str, Any]]) -> int:
        """查询剩余 Pro 轮次（只读，不消耗）。"""
        fp = conversation_fingerprint(messages)
        entry = self._sessions.get(fp)
        if entry is None:
            return 0
        remaining, last_count = entry
        return remaining if len(messages) > last_count else remaining

    # -- 管理 --

    @property
    def active_count(self) -> int:
        """当前活跃的升格对话数。"""
        return len(self._sessions)

    def evict_expired(self) -> int:
        """清理残留的已过期记录。返回 evict 数量。"""
        before = len(self._sessions)
        self._sessions = OrderedDict(
            (k, v) for k, v in self._sessions.items() if v[0] > 0
        )
        return before - len(self._sessions)


# ======================================================================
# Layer 1：启发式复杂度评分
# ======================================================================
# 关键词权重表：每个关键词触发 +0.3，上限 2.0
# 叠加规则：与 token 数、代码块比例、数学符号密度、消息历史长度等其他信号共同计算
# 阈值建议：>= 0.7（或根据 router_threshold 配置）→ 直接升格 Pro
#
# 制定原则：
# 1. 来源：提取自主流 CLI/Router 开源项目的路由逻辑，覆盖所有升格场景类别。
# 2. 双语：每组同时提供简体中文和英文变体，匹配用户的实际输入语言。
# 3. 优先级：按升格信号的明确度降序排列——用户明确要求 > 技术复杂度 > 运营/安全 > 文学/创作。
# 4. 排除多模态：DeepSeek V4 仅支持文本，剔除图像/网页搜索等视觉相关关键词。
# 5. 假阳性过滤：避免过于通用的英文单词（如 plan、derive、performance）单独作为触发词。

_COMPLEXITY_KEYWORDS = [
    # ==================== 1. 用户明确要求提升质量 / 不满 / 重做 ===================
    "重做", "重新生成", "不满意", "不对", "错了", "太差", "受阻", "多次失败",
    "提高质量", "higher quality", "更严谨", "more rigorous", "更好", "better",
    "提升", "improve", "redo with", "更详细", "more detailed",
    "更全面", "comprehensive", "专业", "professional", "专家级", "expert level",
    "stuck", "failing",

    # ==================== 2. 数学 / 证明 ====================
    "证明", "prove", "proof", "proving", "proven",
    "推导", "derivation",
    "定理", "theorem", "引理", "lemma", "推论", "corollary", "公理", "axiom",
    "数学证明", "mathematical proof",
    "求证", "公式推导",
    "复杂度", "complexity", "时间复杂度", "time complexity",
    "空间复杂度", "space complexity", "渐进", "asymptotic",

    # ==================== 3. 架构 / 系统设计 ====================
    "架构", "architecture", "系统设计", "system design",
    "系统架构", "system architecture", "整体设计", "high-level design",
    "微服务", "microservice",
    "分布式", "distributed", "高并发", "concurrency", "可扩展", "scalable",
    "容错", "fault tolerance", "fault-tolerant",
    "容灾", "disaster recovery",
    "一致性", "consistency", "共识算法", "consensus",

    # ==================== 4. 调试 / 排查 ====================
    "crash", "exception", "堆栈", "stack trace", "segfault", "troubleshoot",

    # ==================== 5. 重构 / 优化 ====================
    "重构", "refactor",
    "benchmark", "profiling", "restructure", "rewrite",

    # ==================== 6. 规划 / 推理 / 多步 ====================
    "规划", "计划模式", "计划", "planning", "plan mode",
    "architect", "redesign", "conceptual design", "概念设计",
    "概念", "conceptual", "框架设计", "framework design",
    "多步", "multi-step", "严谨", "rigorous", "理论", "theoretical",
    "逻辑推演", "logical deduction", "假设验证", "hypothesis validation",

    # ==================== 7. 研究 / 调查 / 分析 ====================
    "研究", "调研", "investigate", "research", "深入分析", "in-depth analysis",
    "文献", "literature", "case study", "案例分析",

    # ==================== 8. 算法 / 数据结构 ====================
    "数据结构", "data structure", "复杂算法", "algorithm",
    "算法设计", "algorithm design",

    # ==================== 9. 业务 / 生产级 / 高风险 ====================
    "业务策略", "business strategy", "需求分析", "requirements analysis",
    "生产级", "production-grade", "企业级", "enterprise",
    "迁移", "migrate", "集成", "integrate", "backward compatibility",
    "性能瓶颈", "performance bottleneck", "高风险", "high risk",

    # ==================== 10. 验证 / 确认 / 边界 ====================
    "验证", "validate", "确认", "confirm",
    "edge case", "边界情况", "极端情况", "extreme case",
    "全面测试", "comprehensive test",

    # ==================== 11. 安全 ====================
    "安全审计", "渗透", "XSS", "SQL注入", "privilege escalation",

    # ==================== 12. 多文件 / 大型任务 ====================
    "多文件", "multi-file", "大型代码库", "large codebase",
    "影响范围", "blast radius",

    # ==================== 13. 文学性 / 创意写作 ===================
    "文学性", "文采", "感染力", "修辞", "意境", "生动",
    "更有文采", "提升文笔", "更具感染力", "生动描写", "情感深度", "氛围营造",
    "叙事张力", "人物刻画", "情节发展", "文学表达", "诗意", "诗词",
    "故事写作", "创意写作", "小说创作", "剧本", "散文", "角色扮演", "RP",
    "更有文学性", "文笔更好", "更生动", "意境深远",
    "character development", "emotional impact", "immersive", "storytelling",
    "narrative", "creative prose", "literary style", "more engaging",
]

# 数学 Unicode 符号集合（用于密度检测）
_MATH_SYMBOLS = set("∑∫∂∇∈∉⊂⊃⊆⊇∪∩⇒⇔∀∃≈≡≠≤≥→←↔⟹⟺")


def compute_complexity_score(
    messages: List[Dict[str, Any]],
) -> float:
    """零成本的启发式复杂度评分（Layer 1 快速路径）。

    返回 0.0–10.0 的分数，越高越可能要 Pro。
    各维度加权求和，满分 10。

    通行做法对标：RouteLLM 的请求前静态评分、Cline 的内置复杂度判断。

    2026-04-28：
      - 内容信号（关键词/代码块/数学符号）仅统计 user 消息，排除 system
        消息（如 QWEN.md 项目文档）中的技术词污染。
      - 阈值平滑化：token/轮次/膨胀折扣全部改为线性连续映射。
      - 移除 coding_port 加分（v4-flash 处理简单编码任务效果已极好）。
    """
    if not messages:
        return 0.0

    # 全量文本 → token 估算 + 上下文膨胀分母
    total_text = _flatten_messages(messages)
    estimated_tokens = len(total_text) / 1.8  # 混排 CJK+English
    user_turns = _count_user_messages(messages)

    # user-only 文本 → 内容信号（关键词、代码块、数学符号）
    # system 消息（如 QWEN.md 项目文档）中出现的技术词描述的是项目而非用户请求。
    user_text = _flatten_messages(messages, user_only=True)

    # 1. Token 量（上限 2.5）— 线性连续：8000 tokens → 满分
    token_score = min(estimated_tokens / 3200.0, 2.5)

    # 2. 代码块（上限 2.0）— 线性连续，每个代码块 +0.5（仅 user 消息）
    code_blocks = len(re.findall(r"```", user_text)) // 2
    code_score = min(code_blocks * 0.5, 2.0)

    # 3. 轮次数（上限 2.0）— 线性连续：6 轮 → 满分
    turn_score = min(user_turns / 3.0, 2.0)

    # 4. 关键词命中（上限 2.0）— 线性连续，每个关键词 +0.3（仅 user 消息）
    keyword_hits = sum(user_text.count(kw) for kw in _COMPLEXITY_KEYWORDS)
    keyword_score = min(keyword_hits * 0.3, 2.0)

    # 5. 数学符号（上限 1.5）— 线性连续，每个符号 +0.5（仅 user 消息）
    math_hits = sum(1 for ch in user_text if ch in _MATH_SYMBOLS)
    math_score = min(math_hits * 0.5, 1.5)

    # 6. 上下文膨胀检测 — 平滑折扣（无悬崖效应）。
    #    仅以最近 5 条 user 消息为分母：避免 Coding Agent 注入的巨量
    #    QWEN.md / memory 上下文永久污染比例计算。
    #    折扣因子 d = 0.7 × (1 − fraction / 0.30)，clamp 到 [0, 0.7]。
    #    fraction = 0% → d=0.7（保留 30%）；fraction ≥ 30% → d=0（无折扣）。
    if user_text:
        last_user_msg = ""
        user_msg_lens = []
        for m in reversed(messages):
            if m.get("role") != "user":
                continue
            c = m.get("content", "")
            clen = len(c) if isinstance(c, str) else len(str(c))
            if not last_user_msg:
                last_user_msg = c if isinstance(c, str) else str(c)
            user_msg_lens.append(clen)
            if len(user_msg_lens) >= 5:
                break
        if last_user_msg:
            recent_total = sum(user_msg_lens)
            fraction = len(last_user_msg) / max(recent_total, 1)
            if fraction < 0.30:
                discount = 0.7 * (1.0 - fraction / 0.30)
                token_score *= (1.0 - discount)
                code_score *= (1.0 - discount)

    score = token_score + code_score + turn_score + keyword_score + math_score
    return round(min(score, 10.0), 2)


# ======================================================================
# Sentinel 强制升格（备用入口）
# ======================================================================

_SENTINEL_RE = re.compile(r"<deepproxy_upgrade>\s*force\s*</deepproxy_upgrade>", re.IGNORECASE)


def has_upgrade_sentinel(messages: List[Dict[str, Any]]) -> bool:
    """检查 system prompt 中是否有强制升格标记。

    客户端或上层技能可在 system prompt 中嵌入：
        <deepproxy_upgrade>force</deepproxy_upgrade>
    使当前请求强制升格到 Pro。
    """
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and _SENTINEL_RE.search(c):
                return True
            if isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        if _SENTINEL_RE.search(block.get("text", "")):
                            return True
    return False


# ======================================================================
# 通用工具
# ======================================================================

_EXTRA_BODY_SENTINEL = "_deepproxy_upgrade"


def extra_body_requests_upgrade(body: Dict[str, Any]) -> bool:
    """检查 extra_body 中是否有显式升格请求。

    客户端可发送：
        extra_body={"_deepproxy_upgrade": true}
    该字段在 call_litellm 时被 sentinel 剥离子流程自动移除。
    """
    return bool(body.get(_EXTRA_BODY_SENTINEL, False))


def _flatten_messages(
    messages: List[Dict[str, Any]],
    *,
    user_only: bool = False,
) -> str:
    """将消息列表拼为纯文本（用于评分/分析）。

    Args:
        user_only: 仅提取 role=="user" 的消息，排除 system/assistant。
                   system 消息（如 QWEN.md 项目文档）中出现的"架构""分布式"
                   等词描述的是项目而非用户请求，不应污染复杂度评分。
    """
    parts = []
    for m in messages:
        if user_only and m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)
