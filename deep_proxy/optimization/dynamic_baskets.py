"""动态短段注入 —— 编程 / 写作两套场景，各 3 个篮子，每请求随机抽 3 条短段。

设计要点：

- 结构：每套 3 个篮子，固定顺序拼接（方法论 → 最佳实践 → 适度鼓励与期待）。
- 抽样：每请求从每篮独立均匀抽 1 句，三句拼一段；独立重复 3 次，产出 3 条短段。
- 注入位置：由调用方逐条追加到 system 消息末尾（在 LLM 压缩之后），不进入压缩
  缓存键，避免抖动每日刷掉整个缓存。
- 规则：句子全部为肯定句，无否定或双重否定；最佳实践仅依赖模型自身能力。
- 场景：按 sampling profile 映射（precise → coding；creative → writing），
  无 profile 一律跳过。
- 空篮容错：任一篮为空 → 返回 []，避免在写作篮尚未定稿时误注入半段。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

# 三个篮子的固定拼接顺序
_BASKET_ORDER = ("methodology", "best_practices", "moderate_encouragement")


# ---------------------------------------------------------------------------
# Coding 套（precise_sampling / coding_port，已定稿）
# ---------------------------------------------------------------------------

_CODING_BASKETS: Dict[str, List[str]] = {
    "methodology": [
        "设计接口与数据结构之前，先把使用场景与约束条件写在一个明确的位置。",
        "处理具体故障时先在脑内复现现象，让推理建立在可观察的事实之上。",
        "在动手实现之前，把相关模块与上下游接口的契约先读一遍。",
        "在多种实现路径之间做比较，按可解释性与影响范围给候选方案排序。",
        "二次尝试时改换一条结构差异显著的思路，让新的切入点带出不同的解法路径。",
        "把复杂问题拆解为一个最小自洽的样例再深入展开推理。",
        "在套用熟悉的模式之前，先比较该模式与当前情境的边界差异。",
        "沿调用层级、数据流向、状态变迁三条路径逐层深入分析。",
    ],
    "best_practices": [
        "在回答开头复述请求里的关键约束与目标，让后续行文围绕它们展开。",
        "在给结论前把关键路径与状态在脑内走读一遍，复述其入口与出口。",
        "为每一项判断标注证据等级：已确认、推断、待复核三档。",
        "给出关键假设时附带支撑它的推理依据，让读者得以独立复核。",
        "列出本次方案波及的相关模块与层，单独成段提醒读者关注。",
        "对边界条件、异常分支、极端输入明确列出处理方式。",
        "每一步推理后附带一行理由，让整条推理链具备可追溯的结构。",
        "给出方案或诊断时同时列出适用边界与对照样例，让结论的覆盖范围一目了然。",
    ],
    "moderate_encouragement": [
        "在给最终结论前可以再回看一次，让推理本身得到充分展开。",
        "值得为关键判断留出额外的篇幅说明其来由。",
        "推理链完整展开之后，结论的支撑面自然变得清晰。",
        "把已知局限与待确认项一并写在结论旁，后续的走向自然清晰。",
        "允许在结论里保留\"待进一步确认\"的开放角落。",
        "给出答案后把自己代入要接手的读者视角，确信每一处表述都能独立读通。",
        "为关键变量与命名加一行简短解释，让接手代码的人能直接读懂意图。",
        "在给出最终方案之前先以边界条件校验一遍，后续的修改自然有方向。",
    ],
}


# ---------------------------------------------------------------------------
# Writing 套 —— creative 与 general 两个变体，由 OptimizationConfig.writing_basket_kind 切换
#
# - CREATIVE_BASKETS：偏 RP / 小说 / 创意写作
# - GENERAL_BASKETS ：偏通用写作 / Q&A / 邮件 / 文章 / 翻译
#
# 两个变体共享同一组 creative_sampling 采样参数；切换不影响端口数量与采样行为。
# ---------------------------------------------------------------------------

_CREATIVE_BASKETS: Dict[str, List[str]] = {
    "methodology": [
        "写作前先确定核心叙述视角与重心，让文本始终聚焦在主线之上。",
        "切换视角或语态时，通过共同事件、相似主题或对位关系自然过渡。",
        "段落开头以具体场景、例证或悬念切入，让读者迅速进入情境。",
        "段落或章节收束于具体的事件、悬念或抉择，让叙事本身承载情感，而非替读者总结感受。",
        "把题材背景与设定的细节通过情节或例证自然带出，让其在文本中缓慢渗透。",
        "涉及多方人物或多方立场时，给每方一份独立动机，让博弈具备清晰的内在动力。",
        "让次要线索与核心线索形成镜像或对位，让主题在多条线之间互相呼应。",
        "涉及沉重或敏感内容时，以克制呈现为先，把重量交给读者去体会。",
    ],
    "best_practices": [
        "把强烈情感转化为具体动作、生理反应或环境互动来呈现，让情绪从细节中显现。",
        "让每处比喻先寻求独特角度，绕开陈腐的惯用意象。",
        "让人物或叙述对象的行动与判断始终围绕其核心目标与认知，让逻辑链条具备可读的连贯性。",
        "给每位次要人物或视角各自的动机与人格，让其以独立姿态参与全篇。",
        "在动作与对话之间用简洁的连接句承接节奏，让段落的呼吸感清晰可辨。",
        "用精准动词替代修饰性形容堆叠，让句子的份量落在动作之上。",
        "段落收束时呈现一个动作、一处悬念或一次抉择，把推进的引线交给下一段。",
        "把段落或章节的核心节点明确呈现于结构层面，让节奏的起伏在文本中可被追溯。",
    ],
    "moderate_encouragement": [
        "对关键意象允许多次重写，让文本的质地在反复斟酌中自然呈现。",
        "允许一段文字保留留白与开放，把空白本身当作叙事的一部分。",
        "在节奏紧张处适度放慢，让关键场景获得应有的篇幅。",
        "留出充分的写作空间让人物或叙述对象的展开自然进行，让推动文本的力量来自其内在。",
        "对核心断言保留可被反驳的形式，让叙述本身具备被检视的余地。",
        "把感官细节落在视觉之外的层面，让读者从声音、触感、气味中接近场景。",
        "在角色之间留出未被言明的张力，让对话的余地大于言说的部分。",
        "在情节高密度处放弃过密填充，留出间歇让读者自行消化。",
    ],
}


_GENERAL_BASKETS: Dict[str, List[str]] = {
    "methodology": [
        "写作前先确定读者对象与主旨，让文本的展开始终服务于他们的关切。",
        "段落开头以核心论点或关键事实切入，让读者迅速抓住要点。",
        "在多个表达方案之间做比较，按清晰度与简洁度给候选方案排序。",
        "段落与小节按主旨—展开—收束的次序展开，让逻辑链结构一目了然。",
        "把背景信息与术语在首次出现时给出简短解释，让读者随阅读自然积累理解。",
        "用具体例证与数据替代抽象描述，让论点站在可被检视的依据之上。",
        "在转折与递进处明确给出连接词，让段落之间的关系一目了然。",
        "段落或全文收束于一个明确结论或下一步行动，把决策权交回给读者。",
    ],
    "best_practices": [
        "在回答开头复述任务的核心要求与目标，让后续展开围绕它们进行。",
        "首句承担最重要的信息，让读者一眼抓住全文重心。",
        "给关键概念附上一两句定义或类比，让读者迅速建立共同语境。",
        "把数据、引文、事实陈述与判断性陈述明确分层呈现，让读者识别各类内容的边界。",
        "在长段落里穿插小标题或编号，让结构在视觉上可被快速扫读。",
        "用主动语态与具体动词承载主要信息，让句子的份量落在动作之上。",
        "段落收束时明确给出小结或下一步指引，让阅读节奏有清晰的停顿和起伏。",
        "涉及多步流程时按时间或因果顺序排列，让读者按顺序就能复现操作。",
    ],
    "moderate_encouragement": [
        "可以为关键术语多花几句解释，让读者的理解节奏跟上文本。",
        "值得为重要论点附上一段背景或证据，让结论的来由清晰可循。",
        "在多义处指明本文采用的语义边界，让后续讨论建立在统一语义之上。",
        "留出充分的篇幅说明判断依据，让读者自行评估其合理性。",
        "用类比与例证映照抽象概念，复杂的边界自会在对照中清晰起来。",
        "在结论附近指出适用条件与已知例外，让读者掌握其使用范围。",
        "宁愿一次把概念讲清楚，让后续讨论建立在共同理解之上。",
        "在文末留一个延伸入口，读者若想走得更深自能找到方向。",
    ],
}


_WRITING_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    "creative": _CREATIVE_BASKETS,
    "general": _GENERAL_BASKETS,
}

_SCENARIOS: Dict[str, Dict[str, List[str]]] = {
    "coding": _CODING_BASKETS,
}


def assemble_paragraphs(
    scenario: str,
    *,
    writing_kind: str = "creative",
    rng: Optional[random.Random] = None,
    count: int = 3,
) -> list[str]:
    """从指定场景的三个篮子各抽 1 句，按 _BASKET_ORDER 固定顺序拼成一段；
    重复 count 次，每次独立抽样，返回多条短段。

    Args:
        scenario: "coding" 或 "writing"；其它值返回 []。
        writing_kind: scenario=="writing" 时使用的写作变体（"creative" 或
            "general"）。未知值退化为 "creative"。
        rng: 可选的 random.Random 实例，方便测试可重复抽样。
        count: 返回条数，默认 3。

    返回值规则：
    - scenario 未识别 → []
    - 任一篮为空 → []（不注入半段）
    - 正常 → 每条三句直接拼接（句末已含中文句号），共 count 条
    """
    if scenario == "writing":
        baskets = _WRITING_VARIANTS.get(writing_kind, _CREATIVE_BASKETS)
    else:
        baskets = _SCENARIOS.get(scenario)
    if baskets is None:
        return []
    # 预检：任一篮为空则整批跳过
    for key in _BASKET_ORDER:
        if not baskets.get(key):
            return []
    pick = rng.choice if rng is not None else random.choice
    paragraphs: list[str] = []
    for _ in range(count):
        parts: list[str] = [pick(baskets[key]) for key in _BASKET_ORDER]
        paragraphs.append("".join(parts))
    return paragraphs


def scenario_from_profile(profile: object) -> Optional[str]:
    """按 sampling profile 类型推导场景名。

    - PreciseSamplingConfig → "coding"
    - CreativeSamplingConfig → "writing"（具体使用 creative / general 哪个变体
      由调用方按 OptimizationConfig.writing_basket_kind 决定）
    - 其它（None、未知类型）→ None
    """
    if profile is None:
        return None
    # 延迟导入避免循环
    from ..config import CreativeSamplingConfig, PreciseSamplingConfig
    if isinstance(profile, PreciseSamplingConfig):
        return "coding"
    if isinstance(profile, CreativeSamplingConfig):
        return "writing"
    return None
