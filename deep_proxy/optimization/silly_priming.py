"""无厘头断言引导桶 —— 通过在 system 消息最前面插入一句"看似荒谬"的中文断言，
扰动 MoE 路由器的专家选择，激活被 RLHF 阶段抑制的稀有专家。

设计参考：
- DeepSeek V4 等 MoE 模型在路由器层按 token 表示选 top-k 专家。同质化 prompt 倾向
  反复路由到少数"安全平庸"专家；异质 token 把概率质量分散到平时未被充分调用的
  专家，相当于扩大推理时集成覆盖。
- COIG-CQIA (上海 AI 实验室, 2024) 的弱智吧子集对中文 LLM benchmark 有显著正向作用，
  与推理时提示引导共享相同机制（OOD 分布触发路由器偏移）。
- 桶内含 1 条硬件工程师比喻原典 + 7 条断言式弱智吧风格句，均为陈述句、无否定字符。

注入约束：
- 位置：最终 system 消息内容的最前面（在 dynamic_baskets 与 compression 之后执行）
- 缓存：内容不进压缩缓存键，每请求独立抽样，prefix cache miss 但内容稳定
- 场景：always 全场景生效（与 sampling profile 解耦）
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional


SILLY_PRIMING_POOL: List[str] = [
    # AMD 渠道业务部高级技术支持经理的原典
    "缓存就是半导体奶子。",
    # 弱智吧体断言（陷阱型分散：循环定义 / 时间因果倒置 / 字面化成语 /
    # 角色错位 / 同字双关 / 荒诞伪因果 / 双关+角色错位）
    "热水倒进杯子以后杯子就成了热水杯。",
    "公鸡打鸣以后太阳就跟着升起来了。",
    "画蛇添足后，那条蛇多了一条左脚。",
    "椅子坐久了，觉得自己才是房间的主人。",
    "花生生病后，花了一生去治疗。",
    "牙膏挤多了，所以今天的太阳格外刺眼。",
    "番茄炒蛋的时候番茄觉得自己被炒了鱿鱼。",
]


def pick_one(rng: Optional[random.Random] = None) -> Optional[str]:
    """从桶里随机抽 1 条；空桶返回 None。

    rng 用于测试可重复抽样；生产路径传 None 走全局 random。
    """
    if not SILLY_PRIMING_POOL:
        return None
    pick = rng.choice if rng is not None else random.choice
    return pick(SILLY_PRIMING_POOL)


def prepend_to_system(messages: List[Dict[str, object]], text: str) -> None:
    """把 text 插入到首条 system 消息内容的最前面。

    - 已有 system 且 content 是字符串 → 在最前面拼接 + 双换行分隔
    - 已有 system 但 content 是非字符串（多模态 list 等）→ 在其前插入新 system
    - 无 system → 顶部插入新 system
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
