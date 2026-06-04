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

import re
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Tuple

from ..utils import (
    conversation_fingerprint,
    count_user_messages,
    flatten_messages,
    get_text_from_content,
    last_user_hash,
    last_user_text,
)


# ======================================================================
# Layer 3：对话级持久化 — UpgradeTracker
# ======================================================================
# 通用对话遍历工具（last_user_text / last_user_hash / count_user_messages /
# flatten_messages / conversation_fingerprint）已迁移到 utils.py。
# flash_upgrade 不再自己维护这些 helper。


class RepeatUpgradeThrottle:
    """防刷屏保护：同一对话窗口内同一 user 消息连续触发升格 N 次后，强制降级 Flash + 冷却。

    Coding Agent 场景下同一复杂 prompt 可能被重复提交多次，
    每次 BERT/启发式都会打高分升格到 Pro，造成浪费。

    规则：
      - 同一对话（fingerprint）内同一 user hash 连续触发升格 ≥ max_repeats 次 → 强制 Flash
      - 冷却 cooldown_turns 轮（期间新消息不计数，直接走 Flash）
      - 冷却结束后自动恢复，计数器清零

    键隔离：使用 (conversation_fingerprint, last_user_hash) 组合键，确保
    不同对话窗口的限流状态互不干扰。即使两个窗口有完全相同的最后一条
    user 消息（不同对话中相同的追问），也各自独立计数。

    容量限制：有界 LRU（max_size=2048），与 UpgradeTracker / ReasoningCache 统一策略，
    防止长时间运行时内存无限增长。
    """

    def __init__(self, max_repeats: int = 5, cooldown_turns: int = 3, max_size: int = 2048):
        self._max = max_repeats
        self._cooldown = cooldown_turns
        self._max_size = max_size
        # (fingerprint, last_user_hash) → (consecutive_upgrade_count, cooldown_remaining)
        self._state: OrderedDict[Tuple[str, str], Tuple[int, int]] = OrderedDict()

    def _set(self, key: Tuple[str, str], value: Tuple[int, int]) -> None:
        """写入 key，LRU move_to_end + 驱逐最旧条目。"""
        self._state[key] = value
        self._state.move_to_end(key)
        while len(self._state) > self._max_size:
            self._state.popitem(last=False)

    @property
    def size(self) -> int:
        """当前跟踪的 (fingerprint, last_user_hash) 条目数。供健康检查公开访问。"""
        return len(self._state)

    @property
    def max_repeats(self) -> int:
        """连续触发上限。供日志 / 健康检查 / 决策引擎读取，避免下游穿透 _max。"""
        return self._max

    @property
    def cooldown_turns(self) -> int:
        """冷却轮数。供日志 / 决策引擎读取，避免下游穿透 _cooldown。"""
        return self._cooldown

    def in_cooldown(self, messages: List[Dict[str, Any]]) -> bool:
        """只读查询：当前 (fingerprint, last_user) 是否处于冷却期。

        与 should_throttle 不同，不修改状态、不计数。供 router 在 Step 2
        cache hit 之前预检：若仍在 cooldown，应跳过 cache 直接走 Flash。
        """
        fp = conversation_fingerprint(messages)
        h = last_user_hash(messages)
        entry = self._state.get((fp, h))
        if entry is None:
            return False
        _, cooldown = entry
        return cooldown > 0

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
        fp = conversation_fingerprint(messages)
        h = last_user_hash(messages)
        key = (fp, h)
        entry = self._state.get(key)

        if entry is None:
            self._set(key, (1 if did_upgrade else 0, 0))
            return False

        count, cooldown = entry
        if cooldown > 0:
            # 冷却中：强制 Flash，扣一轮
            self._set(key, (0, cooldown - 1))
            return True

        if did_upgrade:
            count += 1
            if count >= self._max:
                # 冷却 cooldown_turns 轮
                self._set(key, (0, self._cooldown - 1))
                return True
            self._set(key, (count, 0))
        else:
            # 没升格 → 序列中断，归零
            self._set(key, (0, 0))

        return False


class UpgradeTracker:
    """按对话指纹跟踪 Flash→Pro 持久升格状态。

    新轮次 = 最后一条 user 消息变了（hash 变了）。同一轮重试（最后 user 不变）
    不消耗额度；新轮次（最后 user 改变）消耗 1 轮。
    并发安全：不同对话的不同指纹天然隔离。

    为什么用 last_user_hash 而非 len(messages)：
      - len 在 Claude Code 等客户端做对话压缩 / 历史合并时会缩短，导致
        `len_now <= last_len`，counter 永远不递减，对话锁死在 Pro。
      - last user 消息 hash 只反映"是否进入下一轮"，对结构变化稳健。

    存储键为 (fingerprint, last_user_hash, provider)，provider 维度确保同一对话
    在不同 provider（deepseek / mimo）下的升格状态互不干扰。

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
        # key = (fingerprint, last_user_hash, provider)
        # value = remaining_turns
        self._sessions: OrderedDict[Tuple[str, str, str], int] = OrderedDict()
        self._max = max_size

    # -- 公开 API --

    def clear(self, messages: List[Dict[str, Any]], *, provider: str = "deepseek") -> None:
        """主动清除当前对话的升格状态（throttle 触发时同步调用）。

        必要性：throttle 在 router._maybe_upgrade Step 5 触发，但 Step 2
        cache hit 早于 Step 5，下一轮 is_upgraded() 会越过 throttle 直走 Pro。
        清掉 entry 让 throttle 的 cooldown 真正生效。
        """
        fp, last_h = conversation_fingerprint(messages), last_user_hash(messages)
        key = (fp, last_h, provider)
        self._sessions.pop(key, None)

    def is_upgraded(self, messages: List[Dict[str, Any]], *, provider: str = "deepseek") -> bool:
        """当前对话是否处于升格状态。

        副作用：如果这是新轮次（最后 user 消息发生变化），消耗 1 轮剩余额度。
        """
        fp = conversation_fingerprint(messages)
        current_hash = last_user_hash(messages)

        # 先检查当前 last_user_hash 的 key
        key = (fp, current_hash, provider)
        if key in self._sessions:
            remaining = self._sessions[key]
            if remaining <= 0:
                del self._sessions[key]
                return False
            return True

        # 检查是否存在同一 fp + provider 但 last_hash 不同的 entry（新轮次）
        # 遍历所有 key 找匹配 fp + provider
        stale_key = None
        for k in list(self._sessions.keys()):
            if k[0] == fp and k[2] == provider and k[1] != current_hash:
                stale_key = k
                break

        if stale_key is None:
            return False

        remaining = self._sessions[stale_key] - 1  # 新轮次消耗 1
        del self._sessions[stale_key]
        if remaining <= 0:
            return False
        # 写入新的 last_user_hash key
        new_key = (fp, current_hash, provider)
        self._sessions[new_key] = remaining
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return True

    def set_remaining(self, messages: List[Dict[str, Any]], turns: int, *, provider: str = "deepseek") -> None:
        """升格触发后记录剩余 Pro 轮次。

        Args:
            turns: 当前请求之后还能使用 Pro 的轮次数。
                   例如 turns=2 表示当前请求走 Pro + 后续 2 轮。
        """
        self.set_remaining_by_key(
            conversation_fingerprint(messages),
            last_user_hash(messages),
            turns,
            provider=provider,
        )

    def set_remaining_by_key(
        self, fingerprint: str, last_user_hash: str, turns: int, *, provider: str = "deepseek"
    ) -> None:
        """低层入口：用预计算的 fingerprint + last_user_hash 写入。

        用于"延迟提交"场景：决策时（_maybe_upgrade）快照 fp + hash，
        待上游成功后用这两个键提交，避免 messages 在 skills 阶段被改写
        后键失配。
        """
        key = (fingerprint, last_user_hash, provider)
        self._sessions[key] = turns
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    @staticmethod
    def snapshot_keys(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
        """计算 (fingerprint, last_user_hash) 二元组，供延迟提交场景使用。"""
        return conversation_fingerprint(messages), last_user_hash(messages)

    def remaining(self, messages: List[Dict[str, Any]], *, provider: str = "deepseek") -> int:
        """查询剩余 Pro 轮次（只读，不消耗）。"""
        fp = conversation_fingerprint(messages)
        current_hash = last_user_hash(messages)
        key = (fp, current_hash, provider)
        return self._sessions.get(key, 0)

    # -- 管理 --

    @property
    def active_count(self) -> int:
        """当前活跃的升格对话数。"""
        return len(self._sessions)


# ======================================================================
# Layer 1：启发式复杂度评分
# ======================================================================
# 关键词权重表：每个关键词触发 +0.30，user-only 扫描，cap 2.0。
# 与 math / turn / last_user_size / reasoning_density 共同求和，
# 总分 >= heuristic_threshold（默认 8.0）→ 启发式升格 Pro。
#
# 制定原则：
# 1. 来源：提取自主流 CLI/Router 开源项目的路由逻辑，覆盖所有升格场景类别。
# 2. 双语：每组同时提供简体中文和英文变体，匹配用户的实际输入语言。
# 3. 优先级：按升格信号的明确度降序排列——用户明确要求 > 技术复杂度 > 运营/安全 > 文学/创作。
# 4. 排除多模态：当前所有支持的 provider（DeepSeek V4 / MiMo）均为文本-only，剔除图像/网页搜索相关词。
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

ComplexityResult = namedtuple(
    "ComplexityResult", ["score", "user_text", "user_msg_count"]
)


def compute_complexity_score(
    messages: List[Dict[str, Any]],
) -> ComplexityResult:
    """启发式复杂度评分（Layer 1 快速路径）。5 维加权求和上限 10。

    重设记录见 docs/superpowers/specs/2026-05-29-complexity-scoring-redesign.md。
    设计要求：信号在"编码任务"和"用 Claude Code 跑文学创作"两种场景上都需有效。

    维度（均 user-only，除 reasoning_density）：
      1. keyword_score (cap 2.0)        — _COMPLEXITY_KEYWORDS 命中，用户语言意图
      2. math_score    (cap 1.5)        — _MATH_SYMBOLS 密度，数学/形式化任务
      3. turn_score    (cap 2.0)        — user 消息数 / 3，迭代持续度
      4. last_user_size_score (cap 3.0) — 最后一条 user 长度 / 300，当前请求分量
      5. reasoning_score (cap 8.0)      — V4 reasoning_content 最近 3 轮 assistant
                                          平均字符数 / 500，直接测量推理强度。
                                          滑动窗口（非全历史平均）让 Direction C
                                          主动降格在长 agent loop 中可响应——
                                          否则历史深度 reasoning 永久钉高均值。
                                          cap=8.0 (=heuristic_threshold) 让重度
                                          reasoning 单维度即可触发升格——Direction A
                                          唯一触发路径（BERT 仅看 user，看不到 grind）。

    Direction A/B/C 全部由 5 维 + router._maybe_upgrade Step 2 hysteresis 覆盖。
    已删除：token_score（全文本噪声）、code_score（不区分复杂度）、discount
    机制（token_score 删了无需）、agent_depth_score / assistant 双源扫描
    （误判机械与推理同分）。
    """
    if not messages:
        return ComplexityResult(0.0, "", 0)

    user_text = flatten_messages(messages, user_only=True)
    user_turns = count_user_messages(messages)
    last_user = last_user_text(messages)

    # 1. keyword（user-only）— 用户语言意图信号
    keyword_hits = sum(user_text.count(kw) for kw in _COMPLEXITY_KEYWORDS)
    keyword_score = min(keyword_hits * 0.30, 2.0)

    # 2. math（user-only）— 数学 / 形式化任务强信号
    math_hits = sum(1 for ch in user_text if ch in _MATH_SYMBOLS)
    math_score = min(math_hits * 0.50, 1.5)

    # 3. turn — 多轮 = 持续投入（写作迭代 / 多轮 debug）
    turn_score = min(user_turns / 3.0, 2.0)

    # 4. last_user_size — 当前"问题/请求"分量；用最后一条 user 长度而非
    #    全量 token，避免 tool output / CLAUDE.md 注入膨胀的噪声
    last_user_size_score = min(len(last_user) / 300.0, 3.0)

    # 5. reasoning_density — 最近 _REASONING_WINDOW 轮 assistant 的 reasoning
    #    平均长度。直接测量"模型在思考多努力"，跨编码 / 写作都有效；
    #    机械重复任务下 reasoning 几乎为空 → 信号降为零 → 配合 router 侧
    #    downgrade_threshold 形成 Direction C 主动降格。
    #
    #    cap=8.0 = heuristic_threshold：让重度 reasoning 单维度即可触发升格，
    #    覆盖 Direction A（简单 user prompt + 长 agent grind）——这是该
    #    场景唯一的触发路径，因为 BERT 输入 user-only 看不到 grind。
    #
    #    *滑动窗口*（而非全历史平均）至关重要：长 agent loop 早期深度 reasoning
    #    会让全历史平均值永久居高，即便后续转向机械重复也降不下来 → Direction C
    #    永远触发不了（实测：score 钉死在 8-10 直到 hash 改变才会重评估）。
    #    窗口 N=3：3 轮机械化即可让信号降到 ~0；单 chunk 抖动被 3 轮窗口吸收。
    _REASONING_WINDOW = 3
    asst = [m for m in messages if m.get("role") == "assistant"]
    if asst:
        recent = asst[-_REASONING_WINDOW:]
        reasoning_chars = sum(
            len(m.get("reasoning_content") or "") for m in recent
        )
        reasoning_score = min((reasoning_chars / len(recent)) / 500.0, 8.0)
    else:
        reasoning_score = 0.0

    score = (
        keyword_score + math_score + turn_score
        + last_user_size_score + reasoning_score
    )
    return ComplexityResult(round(min(score, 10.0), 2), user_text, user_turns)


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
            text = get_text_from_content(m.get("content", ""))
            if _SENTINEL_RE.search(text):
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
