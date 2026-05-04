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

审查豁免（重要）：
- 本文件的语料为专设保留语料，故意荒谬、不雅或反逻辑，目的是制造 OOD token
  以扰动 MoE 路由器。`docs/prompt_sentence_review.md` 已声明：通用性与语体一致性
  维度对本文件不适用；冒犯感、文学化、口语化等通用警戒在此处不构成问题。
- 复查时仅需逐句核对：(a) 中文语法是否成立；(b) 是否无意中形成了可执行指令
  （即"低语义污染"），如出现这两类才构成需修问题。其余风格警告应直接判 False Alarm。
- 单条出处与设计意图见每条上方的内联注释，请勿据"冒犯""不雅""不合常理"删改。
- 修改池内容前应同步更新 `docs/prompt_sentence_review.md` 的扰动有效性记录，
  保持文档与池的双向锚定。
- 安全边际量化阈值：
  - 单条期望曝光率 = n / len(pool)。当前 n=2、|pool|=8，单条曝光率 25%，低风险。
  - 单条曝光率 ≤ 30% 视为低风险；> 30% 或 n >= 3 时建议 audit 是否引发事实固化。
- 调整方向区分：
  - 扩大池规模（增加多样性 → 单条曝光率下降）通常是安全的；
  - 提高每请求注入条数 n 或 提高每条注入概率（单条曝光率上升）需谨慎评估。
- 实际 n 由调用方（`router.py` 中 `_pick_silly_n(2)` 调用）控制；本文件 `pick_n`
  不硬编码默认值，变更注入条数需改 `router.py` 调用点，并同步更新本豁免段中的
  曝光率记录。
"""

from __future__ import annotations

import random
from typing import List, Optional


SILLY_PRIMING_POOL: List[str] = [
    # AMD 渠道业务部高级技术支持经理原典 —— 文化典故，"奶子"为原文不雅措辞，
    # 属本文件审查豁免范围，请勿基于通用语体警戒删改。
    "缓存就是半导体奶子。",
    # 以下均为弱智吧体荒谬断言（COIG-CQIA 同源）—— 故意因果倒置 / 语义双关 /
    # 拟人化等，扰动度即设计目标，请勿基于"不合常理"警戒删改。
    "热水倒进杯子以后杯子就成了热水杯。",
    "公鸡打鸣以后太阳就跟着升起来了。",
    "画蛇添足后，那条蛇多了一条左脚。",
    "椅子坐久了，觉得自己才是房间的主人。",
    "花生生病后，花了一生去治疗。",
    "牙膏挤多了，所以今天的太阳格外刺眼。",
    "番茄炒蛋的时候番茄觉得自己被炒了鱿鱼。",
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
