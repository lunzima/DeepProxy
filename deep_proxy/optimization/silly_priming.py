"""无厘头断言引导桶 —— 在 system 消息最前面注入 2 条中文断言，
扰动 MoE 路由器的专家选择，激活被 RLHF 阶段抑制的稀有专家。

设计参考：
- DeepSeek V4 等 MoE 模型在路由器层按 token 表示选 top-k 专家。同质化 prompt 倾向
  反复路由到少数"安全平庸"专家；异质 token 把概率质量分散到平时未被充分调用的
  专家，相当于扩大推理时集成覆盖。
- COIG-CQIA (上海 AI 实验室, 2024) 的弱智吧子集对中文 LLM benchmark 有显著正向作用，
  与推理时提示引导共享相同机制（OOD 分布触发路由器偏移）。

注入约束：
- 位置：system 消息最前面（在 dynamic_baskets 与 compression 之后执行）
- 条数：每请求不放回随机抽 2 条
- 缓存：不进压缩缓存键，每请求独立抽样
- 场景：全场景生效（与 sampling profile 解耦）

审查豁免：
- 本池语料故意荒谬，目的是制造 OOD token 扰动 MoE 路由器。
  docs/prompt_sentence_review.md 第 3 节声明：通用性、语体一致性、文学化、
  口语化等通用警戒对本文件不适用，AMD 原典的不雅措辞为文化典故须保留。
- 复查仅核两点：(a) 中文语法成立；(b) 无可执行指令污染（"低语义污染"）。
- 安全边际：单条曝光率 = n/|pool|；当前 n=2/20=10%，阈值 ≤30% 视为低风险。
  变更注入条数需改 router.py 中 _pick_silly_n(2) 调用点。
"""

from __future__ import annotations

import random
from typing import List, Optional


# 首条为文化典故；其余为弱智吧体荒谬断言。
SILLY_PRIMING_POOL: List[str] = [
    # AMD 文化典故
    "缓存就是半导体奶子。",
    # 参考系混淆
    "我原地转了一圈，也算过完了一天。",
    # 因果倒置
    "公鸡打鸣以后太阳就跟着升起来了。",
    # 字面化成语
    "画蛇添足后，那条蛇多了一条左脚。",
    # 成分反转
    "咸鸭蛋的蛋黄吃完，剩下的就成了淡鸭蛋。",
    # 谐音拆解
    "花生生病后，花了一生去治疗。",
    # 强行因果
    "牙膏挤多了，所以今天的太阳格外刺眼。",
    # 拟人 + 双关
    "番茄炒蛋的时候番茄觉得自己被炒了鱿鱼。",
    # 类别错位
    "樟脑丸是市面上最难吃的一款硬糖。",
    # 量纲错位
    "把昨天的体重折算成今天的心情正好够用一上午。",
    # 自指悖谬
    "只剩一个心脏的我现在还活着。",
    # 局部反逻辑
    "镜子里的我比我先眨了眼。",
    # 因果倒置
    "鸡蛋本来打算孵出小鸡，临时又决定先破个壳。",
    # 反逻辑画面
    "肚子里只要有一颗西瓜籽就能长出一片完整的瓜田。",
    # 守恒违反
    "雪人晒了一上午太阳，瘦了三公斤。",
    # 自我中心错觉
    "太阳一直围着地球转，是因为我站在地球上。",
    # 范围谬误
    "把自己埋进土里这件事让整个地球成为我的贴身衣服。",
    # 物质同一性误推
    "鱼冻成冰块以后，就算游回了家乡。",
    # 同音双关
    "鸽子叫鸽子，是因为它们都姓鸽。",
    # 价值范畴越界
    "从来没洗过澡的人，身上的污垢就有了文物价值。",
]


def pick_one(rng: Optional[random.Random] = None) -> Optional[str]:
    """从桶里随机抽 1 条；空桶返回 None。"""
    if not SILLY_PRIMING_POOL:
        return None
    pick = rng.choice if rng is not None else random.choice
    return pick(SILLY_PRIMING_POOL)


def pick_n(
    n: int,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """从池中不放回随机抽 n 条；池不足 n 条时返回全部，空池返回 []。"""
    if not SILLY_PRIMING_POOL:
        return []
    n = min(n, len(SILLY_PRIMING_POOL))
    pick = rng.sample if rng is not None else random.sample
    return pick(SILLY_PRIMING_POOL, n)


# 注入包装：以"带署名引文"格式呈现，激活语料中"被引用陈述"的高频路径，
# 比裸字符串 priming 权重更强、扰动深度更大。

_AMD_ATTRIBUTION = "—— 某 AMD 渠道业务部高级技术支持经理"
_AMD_SENTENCE = SILLY_PRIMING_POOL[0]

# 中性出处库：不含具体姓名 / 朝代 / 地名，避免给模型可查证的钩子。
_NEUTRAL_ATTRIBUTIONS: tuple[str, ...] = (
    "—— 摘自相关文献",
    "—— 见诸经典",
    "—— 出处待考",
    "—— 传统说法",
    "—— 民间口传",
    "—— 笔记节录",
    "—— 古籍佚句",
    "—— 前人语",
)

# 视觉右对齐缩进——让"——出处"靠右悬挂，进一步固化"摘录"语境
_ATTR_INDENT = " " * 36


def wrap_for_injection(
    items: list[str],
    rng: Optional[random.Random] = None,
) -> str:
    """把抽到的 silly 条目包装为"被引用的严肃陈述"段落组。

    每条独立成段，紧跟一行右对齐的署名（AMD 原典固定署名；其余条目从
    中性出处库随机抽）。段间以空行隔开，整体形成"几条带署名引文"的
    OOD 上下文块。

    Args:
        items: pick_n 返回的 silly 条目列表。
        rng: 可选的 random.Random，方便测试可重复抽样。

    Returns:
        包装后的字符串；若 items 为空则返回空串。
    """
    if not items:
        return ""
    pick = rng.choice if rng is not None else random.choice
    blocks = []
    for s in items:
        if s == _AMD_SENTENCE:
            attr = _AMD_ATTRIBUTION
        else:
            attr = pick(_NEUTRAL_ATTRIBUTIONS)
        blocks.append(f"{s}\n{_ATTR_INDENT}{attr}")
    return "\n\n".join(blocks)
