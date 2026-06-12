"""响应侧风格后处理 — 纯 regex + 字符串操作，零 LLM 调用。

对上游生成的中文小说文本做机械扫描，命中后向会话注入用户反馈消息，
让 LLM 自行修正。不执行自动替换（避免机械替换造成质量损失）。

设计约束：
  - 全部为同步纯函数，无 I/O，<1ms 延迟
  - 单一 on/off 开关，不逐条启用/禁用规则
  - 反馈使用陈述判断句式（「XX是禁用的写作模式」），非祈使命令
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规则数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleRule:
    """单条风格检测规则。"""

    id: str  # 唯一标识
    pattern_name: str  # 模式名称，写入反馈消息
    pattern: re.Pattern  # 已编译的正则
    example_fix: str  # 正面改写指引
    quote_violation: bool = True  # 反馈中是否引用违规原文。False 时仅使用 pattern_name，
    # 避免高频模式被引用后反而被 LLM 学习强化（"禁词表当提词器"效应）。
    # 高风险模式（归因句式、套路收尾、抽象词标签）设为 False。


def _compile(pattern: str) -> re.Pattern:
    """将 grep-style 管道分隔的模式编译为 case-insensitive regex。"""
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# A 线：人体解剖学 + 物理测量（QWEN.md §2.7）
# ---------------------------------------------------------------------------

A1 = StyleRule(
    "a1", "人体解剖学词汇",
    _compile(r"骨架|骨骼|锁骨.*前端|肩峰|桡骨|尺神经|跖骨|骶骨|髋骨"),
    "改为日常用词：「后背」「全身」「肩膀」",
)

A2 = StyleRule(
    "a2", "皮肤底下 / 皮肤下面",
    _compile(r"皮肤底下|皮肤下面"),
    "改为表面描写",
)

A3 = StyleRule(
    "a3", "一截（阴影/月光）",
    _compile(r"一截.*阴影|阴影.*一截|一截.*月光|月光.*一截"),
    "直接描写位置变化，如「阴影移了过去」",
)

A4 = StyleRule(
    "a4", "夹住空气 / 画圈",
    _compile(r"夹住了.*空气|排空.*空气|空气.*收缩|画了一个.*圈|画了.*半个圈|在.*描了一"),
    "用具体动作替代，如「门板合上，闷响了一声」",
)

A5 = StyleRule(
    "a5", "半寸 / 一息 / 几息（测量式口癖）",
    _compile(r"半寸|了三息|了几息|了一息|过了数息|几息之后|数息之间|一息之间"),
    "动词重叠式（「推了推」）或具体动作（「他把茶杯转了一圈」）",
)

# ---------------------------------------------------------------------------
# B 线：声音物理化（QWEN.md §2.7）
# ---------------------------------------------------------------------------

B1 = StyleRule(
    "b1", "声音被物理力处理（吞/压/切/收/挤/散）",
    _compile(r"被.*吞掉|声音.*吞|吞掉.*音|吞掉.*声|声音.*压缩|被压缩.*声"
             r"|声音.*切断|切成.*段.*声|声音.*收住|被.*壁.*收住"
             r"|声音.*散尽|余音.*散|声响.*散掉|声音.*散掉"),
    "改为终止动词：「声音远了」「安静下来」「越来越轻」",
)

# ---------------------------------------------------------------------------
# C 线：泛物理吞没（QWEN.md §2.7）
# ---------------------------------------------------------------------------

C1 = StyleRule(
    "c1", "抽象概念被物理力作用（冲散/淹没/吞噬/压缩）",
    _compile(r"被.*冲散|被.*淹没|被.*吞噬|被.*压缩"),
    "改为抽象动词：「暗淡下去」「消散」「变安静」",
)

# ---------------------------------------------------------------------------
# D 线：模糊描述口癖（QWEN.md §2.7）
# ---------------------------------------------------------------------------

D1 = StyleRule(
    "d1", "模糊指代（某种/一种/东西）",
    _compile(r"某种.*东西|一种.*东西|有.*东西|某种.*的.*困惑|某种.*的.*感觉|某种.*的.*气息"),
    "用具体感官细节替代",
)

D2 = StyleRule(
    "d2", "模糊描述（说不上来/看不懂）",
    _compile(r"说不上来|说不清楚|说不清"),
    "明确写出具体内容",
)

# ---------------------------------------------------------------------------
# E 线：物品拟人化（QWEN.md §2.7）
# ---------------------------------------------------------------------------

E1 = StyleRule(
    "e1", "物品被赋予人的意图（提醒/告诉/欢迎/拒绝）",
    _compile(r"建筑.*提醒|大楼.*提醒|房子.*提醒|沉默.*告诉|眼神.*告诉"
             r"|椅子.*欢迎|桌子.*欢迎|门.*欢迎"
             r"|沉默.*拒绝|空气.*拒绝|时间.*拒绝"),
    "写物品的物理特征",
)

# ---------------------------------------------------------------------------
# F 线：POV 模板化（QWEN.md §2.7）
# ---------------------------------------------------------------------------

F1 = StyleRule(
    "f1", "目光/余光扫到",
    _compile(r"目光.*扫到|余光.*扫到"),
    "交替切入方式：听觉、触觉、对话",
)

F2 = StyleRule(
    "f2", "注意到 / 心想",
    _compile(r"注意到|心想"),
    "交替切入方式：听觉、触觉、对话",
)

# ---------------------------------------------------------------------------
# G 线：体制知识展示（QWEN.md §2.7）
# ---------------------------------------------------------------------------

G1 = StyleRule(
    "g1", "体制化表述（机关里浸/公门磨）",
    _compile(r"机关里.*浸|体制内.*浸|公门.*磨|久历公门|浸过几年|磨出来的沉"),
    "简化为具体判断：「步态沉稳」",
)

# ---------------------------------------------------------------------------
# H 线：感知精确绑定（QWEN.md §2.7）
# ---------------------------------------------------------------------------

H1 = StyleRule(
    "h1", "均匀/精密/恰好",
    _compile(r"均匀的.*凹陷|均匀的.*痕迹|均匀的.*磨损|精密计算|恰好是|高度恰好"),
    "改为 POV 视角内的判断",
)

# ---------------------------------------------------------------------------
# I 线：生涩语言（QWEN.md §2.7）
# ---------------------------------------------------------------------------

I1 = StyleRule(
    "i1", "三「的」密集嵌套（of 属格链直译）",
    _compile(r"的[^，。！？、]{0,6}的[^，。！？、]{0,6}的"),
    "拆句或调整语序",
)

I2 = StyleRule(
    "i2", "形式主语残留（这使/这令/这让）",
    _compile(r"这使|这令|这让"),
    "删除形式主语，让真实主语直接连接动词",
)

I3 = StyleRule(
    "i3", "「当……时」从句滥用",
    _compile(r"当[^，。；]{1,20}时[,，]"),
    "改为「一……就……」或直接衔接",
)

# ---------------------------------------------------------------------------
# §3.4 尾音描写禁用（QWEN.md）
# ---------------------------------------------------------------------------

TAIL = StyleRule(
    "tail", "尾音微观描写",
    _compile(r"尾音.*飘|尾音.*沉|尾音.*翘|尾音.*掉|尾音.*颤|尾音.*消散"
             r"|尾音.*拖长|尾音.*藏着|尾音.*带着|尾音.*透着"
             r"|尾音.*拖得略长|每个字的尾音都"),
    "用动作或停顿传递语气",
)

# ---------------------------------------------------------------------------
# §3.4.2 身体动作的逐帧拆解禁用（QWEN.md）
# ---------------------------------------------------------------------------

FR1 = StyleRule(
    "fr1", "手指逐帧拆解",
    _compile(r"两个拇指|十根手指.*弯曲|十根手指.*张开|指腹.*按进"
             r"|拇指.*指节.*蹭|五根手指.*一根|五指并拢|手指从半握"),
    "整体动作：「双手搁在桌面」「攥紧了手」",
)

FR2 = StyleRule(
    "fr2", "身体关节序列",
    _compile(r"从肩膀到颈椎|一条腿.*跨.*另一条|上半身.*探出|手臂.*勾住|手腕.*反向.*拧"),
    "整体动作：「点了点头」「他扑向窗口」",
)

FR3 = StyleRule(
    "fr3", "下颌/牙关微动作",
    _compile(r"下颌.*绷了一下|牙关.*咬.*松开|齿关.*咬"),
    "改为整体表情描写",
)

FR4 = StyleRule(
    "fr4", "眼球/瞳孔微观追踪",
    _compile(r"瞳孔.*收缩|瞳孔.*聚焦|瞳孔.*焦距|眼珠.*追着|瞳孔.*光.*收拢"),
    "改为「目光变了」「神色一紧」",
)

FR5 = StyleRule(
    "fr5", "否定式描写（以缺席动作替代在场）",
    _compile(r"没有回头|没有说话|没有动[^静O]|没有回答|没有接话|没有出声|没有开口"),
    "写出此刻的在场动作",
)

# ---------------------------------------------------------------------------
# §3.1 破折号禁用（QWEN.md / 通用创作要求.md §3.1）
# ---------------------------------------------------------------------------

DASH = StyleRule(
    "dash", "叙事文本中的破折号",
    _compile(r"——"),
    "用逗号、句号或分号替代",
)

# ---------------------------------------------------------------------------
# §3.2 归因句式禁用（QWEN.md / 通用创作要求.md §3.4）
# ---------------------------------------------------------------------------

ATTR_A = StyleRule(
    "attr_a", "「不是…而是…」/「不是…是…」/「并非…而是…」归因句式",
    _compile(r"不是.*而?是|并非.*而是"),
    "直接写行为，读者自行判断",
    quote_violation=False,
)

ATTR_B = StyleRule(
    "attr_b", "叙事评论式机制解读（本性/天性/宿命/本质）",
    _compile(r"本性|天性|宿命|本质"),
    "写具体行为，不归结到抽象本质",
    quote_violation=False,
)

# ---------------------------------------------------------------------------
# 通用创作要求.md 补充
# ---------------------------------------------------------------------------

CLIC = StyleRule(
    "clic", "陈腐表达",
    _compile(r"投入湖面的石子|指节泛白|心中一紧|五味杂陈"),
    "用具体、独特的描写替代",
    quote_violation=False,
)

ABST = StyleRule(
    "abst", "空洞浪漫化抽象词",
    _compile(r"岁月|时光|温柔|温暖|远方|梦想|灵魂"),
    "通过具体细节承载情感",
    quote_violation=False,
)

EMO = StyleRule(
    "emo", "疼痛文学情感标签",
    _compile(r"孤独|破碎|敏感|受伤"),
    "写具体行为替代情感命名",
    quote_violation=False,
)

# ---------------------------------------------------------------------------
# DeepSeek V4 用户反馈高频句式补充
# 来源：github.com/victorchen96/deepseek_v4_rolepaly_instruct
# 861 条结构化 badcase，471 位用户反馈（20260526）
# ---------------------------------------------------------------------------

RP1 = StyleRule(
    "rp1", "套路化收尾（这就够了/那就够了）",
    _compile(r"这[就是]够了|那[就是]够了"),
    "用具体动作或场景收尾，而非抽象安慰",
    quote_violation=False,
)

RP2 = StyleRule(
    "rp2", "AI 情感动词（稳稳接住/兜住情绪）",
    _compile(r"稳稳接住|稳稳地接住|兜住.*情绪|兜住.*感情|接住.*情绪|轻轻落下"),
    "用具体的身体反应或对话替代抽象情感动词",
    quote_violation=False,
)

RP3 = StyleRule(
    "rp3", "总结性套话（总而言之/综上所述）",
    _compile(r"总而言之|综上所述"),
    "删除总结句，让行为自行传递信息",
)

RP4 = StyleRule(
    "rp4", "议论文模板（是…的基石/必修课/必经之路）",
    _compile(r"是.*的基石|是.*的必修课|是.*的必经之路|是.*的关键所在"),
    "用具体行为替代抽象判断",
    quote_violation=False,
)

RP5 = StyleRule(
    "rp5", "固定微表情（眨了眨眼/喉结滚动）",
    _compile(r"眨了眨眼|喉结滚动|喉结上下滚动"),
    "用角色特有的反应方式替代通用微表情模板",
)

RP6 = StyleRule(
    "rp6", "模板比喻（像在说今天的天气）",
    _compile(r"像在说.*天气|语气平淡得仿佛|语气平淡得像在"),
    "用具体的感官细节替代抽象比喻",
    quote_violation=False,
)

RP7 = StyleRule(
    "rp7", "三连排比否定（不X，不Y，不Z）",
    _compile(r"不[^，。；]{1,4}[， ]+不[^，。；]{1,4}[， ]+不"),
    "用具体描写替代排比否定句式",
    quote_violation=False,
)

RP8 = StyleRule(
    "rp8", "矛盾式表达（很…，但很…）",
    _compile(r"很[^，。；]{0,4}[， ]+但很"),
    "用具体事实替代抽象矛盾判断",
    quote_violation=False,
)

RP9 = StyleRule(
    "rp9", "强行升华收尾（或许这就是/大概这就是）",
    _compile(r"或许这就|大概这就|也许这就|或许这便"),
    "用具体动作或事实收束，不替读者完成情感归纳",
    quote_violation=False,
)

RP10 = StyleRule(
    "rp10", "「无X无Y」格式化双否定（无风无浪/无悲无喜）",
    _compile(r"无[^，。；]{0,3}无"),
    "用具体描写替代格式化成语",
    quote_violation=False,
)

RP11 = StyleRule(
    "rp11", "「之所以……是因为……」上帝视角因果链",
    _compile(r"之所以.*是因为"),
    "行为自身承载信息，不通过因果链替读者做总结",
    quote_violation=False,
)

# ---------------------------------------------------------------------------

RULES: list[StyleRule] = [
    A1, A2, A3, A4, A5,
    B1,
    C1,
    D1, D2,
    E1,
    F1, F2,
    G1,
    H1,
    I1, I2, I3,
    TAIL,
    FR1, FR2, FR3, FR4, FR5,
    DASH,
    ATTR_A, ATTR_B,
    CLIC, ABST, EMO,
    RP1, RP2, RP3, RP4, RP5,
    RP6, RP7, RP8, RP9, RP10, RP11,
]

# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------


def _extract_sentence(text: str, match_start: int, match_end: int) -> str:
    """提取匹配所在的完整句子（以句号/问号/感叹号/换行切割）。"""
    # 向前找句首
    start = match_start
    while start > 0 and text[start - 1] not in "。！？\n":
        start -= 1
    # 向后找句尾
    end = match_end
    while end < len(text) and text[end] not in "。！？\n":
        end += 1
    if end < len(text) and text[end] in "。！？":
        end += 1  # 包含句尾标点
    return text[start:end].strip()


def scan_violations(text: str, rules: list[StyleRule] | None = None) -> list[dict]:
    """对文本扫描所有规则，返回命中列表。

    每条命中: {
        "rule_id": str,
        "pattern_name": str,
        "match_text": str,   # 正则命中的子串
        "sentence": str,     # 命中所处的完整句子
        "example_fix": str,
        "quote_violation": bool,  # 反馈中是否引用违规原文
    }

    按 rule_id (rule.id) 去重：同一规则在同一句子中只保留第一次命中。
    """
    if rules is None:
        rules = RULES

    hits: list[dict] = []
    seen: set[tuple[str, int]] = set()  # (rule_id, sentence_start)

    for rule in rules:
        for m in rule.pattern.finditer(text):
            sentence = _extract_sentence(text, m.start(), m.end())
            key = (rule.id, text.find(sentence))  # sentence start position
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "rule_id": rule.id,
                "pattern_name": rule.pattern_name,
                "match_text": m.group(),
                "sentence": sentence,
                "example_fix": rule.example_fix,
                "quote_violation": rule.quote_violation,
            })

    return hits


# ---------------------------------------------------------------------------
# 反馈构造
# ---------------------------------------------------------------------------

_FEEDBACK_PREAMBLE = (
    "叙事风格自检反馈：\n\n"
    "下列描写触及了禁用的写作模式：\n"
)

_FEEDBACK_POSTAMBLE = "\n请修正以上内容后重新提交。"


def build_feedback_message(violations: list[dict]) -> str:
    """构造陈述判断式反馈消息（「XX是禁用的写作模式」+ 正面改写指引）。

    对 quote_violation=False 的高风险模式（归因句式、套路收尾、抽象词标签等），
    不在反馈中引用违规原文（避免 "禁词表当提词器" 效应），仅使用模式名称。
    """
    lines = [_FEEDBACK_PREAMBLE]
    for i, v in enumerate(violations, 1):
        if v.get("quote_violation", True):
            # 正常模式：引用违规原文 + 模式判断 + 改写指引
            lines.append(
                f'{i}. 「{v["sentence"]}」\n'
                f'   {v["pattern_name"]}是禁用的写作模式。{v["example_fix"]}。\n'
            )
        else:
            # 高风险模式：仅使用模式名称，不引用原文（防止负面表达注入强化）
            lines.append(
                f'{i}. {v["pattern_name"]}是禁用的写作模式。{v["example_fix"]}。\n'
            )
    lines.append(_FEEDBACK_POSTAMBLE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 重发循环（供 router 调用）
# ---------------------------------------------------------------------------

async def apply_style_guard_loop(
    body: dict,
    call_upstream: Callable,
    result: dict,
    rules: list[StyleRule] | None = None,
    max_retries: int = 4,
) -> dict:
    """扫描 assistant 响应文本 → 反馈 → 重发上游 循环。

    每次重发时，将前一轮的 assistant 消息和风格反馈用户消息附加到
    body["messages"] 中，重新调用 call_upstream。

    返回最终的 response dict（无违规时直接返回原 result）。
    """
    if rules is None:
        rules = RULES

    for _retry in range(max_retries):
        # 提取 assistant 文本内容
        choices = result.get("choices", [])
        if not choices:
            break
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            break

        violations = scan_violations(content, rules)
        if not violations:
            break

        feedback = build_feedback_message(violations)
        logger.info("style_guard retry=%d violations=%d rule_ids=%s",
                     _retry + 1, len(violations),
                     sorted(set(v["rule_id"] for v in violations)))

        # 附加前一轮的 assistant 消息和反馈
        body["messages"].append(choices[0]["message"])
        body["messages"].append({"role": "user", "content": feedback})

        result = await call_upstream()

    return result
