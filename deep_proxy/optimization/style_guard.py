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
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# 告警日志目录（与主日志同目录）
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_ALERT_LOG = _LOG_DIR / "style_guard_alerts.log"
try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    logger.warning("无法创建 StyleGuard 告警日志目录: %s", _LOG_DIR)

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
    _compile(r"骨架|骨骼|锁骨.*前端|肩峰|桡骨|尺神经|跖骨|骶骨|髋骨"
             r"|肩胛骨|脊椎|肌腱|髂骨"),
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
             r"|声音.*散尽|余音.*散|声响.*散掉|声音.*散掉"
             r"|从晶石.*挤出来|声音.*挤出来"),
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
    _compile(r"说不上来|说不清楚|说不清"
             r"|看不懂的东西|看不懂.*什么|有种说不上来"),
    "明确写出具体内容",
)

# ---------------------------------------------------------------------------
# E 线：物品拟人化（QWEN.md §2.7）
# ---------------------------------------------------------------------------

E1 = StyleRule(
    "e1", "物品被赋予人的意图（提醒/告诉/欢迎/拒绝）",
    _compile(r"建筑.*提醒|大楼.*提醒|房子.*提醒|沉默.*告诉|眼神.*告诉"
             r"|椅子.*欢迎|桌子.*欢迎|门.*欢迎"
             r"|沉默.*拒绝|空气.*拒绝|时间.*拒绝"
             r"|包浆.*告诉|墙壁.*告诉|灯光.*告诉"),
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
    _compile(
        r"没有(?:看[^到见过]|说[^话明]|问[^题号]|走|来|去|做|动[^作物静]|笑|哭|回[^到来]|接[^着受]|出[^来现声]|放|拿|给|坐|站|喝|吃|碰|摸|拍|敲|打|骂|拉|推|踢|踩|瞪|躲|挡|扶|拎|抱|背|扛|搬|拖|插|掏"
        r"|开口|出声|回头|转身|抬头|低头|伸手|动手|还手|吭声|吱声|应声|理[^由解]"
        r"|提[^醒供前]|叫[^做]"  # 没提、没叫人
        r"|人|话|声|应"  # 没有人、没有说话、没有出声、没有回应
        r")"
        r"|(?:(?:也|[，。；])(?:就|偏|硬|愣|却)[没不][一-鿿])"  # 他没走、他不看
        r"|[没不][一-鿿][，。；！？]"  # 不咳、没应
        r"|[没不][一-鿿]{2}(?!差[不多]|到[了]|多[久]|知[道]|看[见]|见[过]|关[系])",
    ),
    "写出此刻的在场动作：坐他在做什么、看他在看什么、说他开口说了什么",
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
# 时间计数式微停顿（QWEN.md §2.6 口癖表）
# ---------------------------------------------------------------------------

TM1 = StyleRule(
    "tm1", "时间计数式微停顿（停了一下/片刻/几息）",
    _compile(
        r"停了一下"
        r"|顿了一下"
        r"|停了片刻|顿了片刻|站了片刻|坐了片刻|沉默了片刻"
        r"|几息(?![间之])"  # 排除"几息之间"
        r"|数息|半息|三息"
    ),
    "用一个具体的知觉或动作替代空白停顿。写出这段时间里人物看见了什么、听见了什么、手指触碰到了什么。",
)

# ---------------------------------------------------------------------------
# 空格填充式场景收束（每段独立匹配，OR逻辑）
# ---------------------------------------------------------------------------

SE1 = StyleRule(
    "se1", "空格填充式场景收束（安静了很久/沉默了很久）",
    _compile(
        r"安静了很长时间"
        r"|沉默了很长时间"
        r"|安静了很久"
        r"|安静了下来"
    ),
    "用在场行为收束场景：人物的一个动作或环境的一声响动替代静默宣告。",
)

# ---------------------------------------------------------------------------
# 静态保持/维持（V4 冻结帧模式）
# ---------------------------------------------------------------------------

INV1 = StyleRule(
    "inv1", "保持/维持着静止物理状态（冻结帧）",
    _compile(
        r"(?:保持|维持)着.{0,6}(?:角度|姿势|弧度|轮廓|节奏|原状|位置)"
    ),
    "写出人物正在做什么具体动作。动作本身会自然带出身体位置。不把人物写成静止的雕塑。",
    quote_violation=False,
)

# ---------------------------------------------------------------------------
# QWEN.md 遗漏补充：G2 (引号错配) / G9 (的情况下) / G13 (虚义它) / G15 (叙事者总结)
# ---------------------------------------------------------------------------

QUOTE_MISMATCH = StyleRule(
    "q_quote", "引号字符错配：冒号后误跟了右双引号(U+201D)",
    _compile(r"”"),
    "所有冒号(:)后的左起位必须使用左引号U+201C，而非右引号U+201D。请在编辑时从原文已正确使用的对话中复制现成的左引号，再粘贴到目标位置。",
    quote_violation=False,
)

Q_TRANSLATIONESE_DE = StyleRule(
    "q_de", "翻译腔「在……的情况下」从句",
    _compile(r"在.{1,20}的情况下"),
    "用自然的中文时序词替代：「如果……」「当……」或直接说出结果。例：'在没有通知的情况下'→'事先没有通知'。",
)

Q_DUMMY_IT = StyleRule(
    "q_it", "虚义「它」/「这使」/「这让」作形式主语",
    _compile(r"(?:这使|这令|这让|那使|那令)[^步]"  # 排除"这让步"等固定搭配
             r"|[。；]它让|，它让|[。；]它使|[。；]它令"),
    "删除形式主语，让真实主语直接连接谓语动词。例：'这让想起'→'想起'。",
)

Q_NARRATOR_CONCLUSION = StyleRule(
    "q_narrator", "叙事者总结句式（这意味着/这就是/那就是/这便）",
    _compile(r"这意味着|这就是|那就是|这便意味着|这便|这便是|那便是"
             r"|腐败的代价|这就是.*的代价"),
    "删除叙事者总结句，用具体动作或感官细节收束。例：'这意味着……'→直接写人物做了什么。",
    quote_violation=False,
)

# ---------------------------------------------------------------------------
# 显式跳过标签
# ---------------------------------------------------------------------------

STYLE_OVERRIDE_TAG = "[style-override]"


def _strip_override_tag(content: str) -> str:
    """移除跳过标签（行内嵌入或独立一行）。"""
    return content.replace(STYLE_OVERRIDE_TAG, "")


def _has_override_tag(content: str) -> bool:
    """检查 assistant 响应是否含有显式跳过标签。"""
    return STYLE_OVERRIDE_TAG in content


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
    TM1, SE1, INV1,
    QUOTE_MISMATCH, Q_TRANSLATIONESE_DE, Q_DUMMY_IT, Q_NARRATOR_CONCLUSION,
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


# ---------------------------------------------------------------------------
# 段落级扫描辅助函数（G1 连续短句 / G3「的时候」密度 / G11 比喻词密度）
# ---------------------------------------------------------------------------

_SHORT_SENTENCE_LIMIT = 10   # 短句字符数上限
_SHORT_CHAIN_MIN = 3          # 连续短句最低触发数
_DESHIHOU_LIMIT = 3           # 单段「的时候」触发数
_METAPHOR_LIMIT = 3           # 单段「仿佛/似乎/好像」触发数
_DIALOGUE_CHARS = {'"', '"', '「', '」', ':', '…'}  # 对话锚点字符


def _split_paragraphs(text: str) -> list[str]:
    """按空白行将文本切分为段落。"""
    return re.split(r'\n\s*\n', text)


def _split_sentences(paragraph: str) -> list[str]:
    """将段落按句末标点切分为句子列表。"""
    sentences = re.split(r'(?<=[。！？])', paragraph)
    return [s.strip() for s in sentences if s.strip()]


def _is_dialogue_sentence(sent: str) -> bool:
    """判断句子是否为对话（含引号或以引号开头）。"""
    if any(ch in sent for ch in _DIALOGUE_CHARS):
        return True
    if sent.startswith('"') or sent.startswith('"') or sent.startswith('「'):
        return True
    return False


def _detect_short_sentence_chains(text: str) -> list[dict]:
    """G1：检测连续 ≥3 句短句（≤10字）的非对话序列。

    返回命中列表，每条含 paragraph 级别的上下文。
    """
    hits: list[dict] = []
    paragraphs = _split_paragraphs(text)
    for para_idx, para in enumerate(paragraphs):
        sentences = _split_sentences(para)
        if len(sentences) < _SHORT_CHAIN_MIN:
            continue
        for i in range(len(sentences) - _SHORT_CHAIN_MIN + 1):
            window = sentences[i:i + _SHORT_CHAIN_MIN]
            # 全部 ≤10字 且全部非对话
            if all(len(s) <= _SHORT_SENTENCE_LIMIT for s in window) and \
               not any(_is_dialogue_sentence(s) for s in window):
                # 扩展窗口
                j = i + _SHORT_CHAIN_MIN
                while j < len(sentences) and len(sentences[j]) <= _SHORT_SENTENCE_LIMIT \
                        and not _is_dialogue_sentence(sentences[j]):
                    j += 1
                chain = sentences[i:j]
                hits.append({
                    "rule_id": "g1_short",
                    "pattern_name": f"连续{j - i}个短句（均≤{_SHORT_SENTENCE_LIMIT}字）",
                    "match_text": " ".join(chain),
                    "sentence": "".join(chain),
                    "example_fix": "将短句合并为语义完整的复合句，用逗号或分号连接。",
                    "quote_violation": True,
                })
                break  # 每段只报一次
    return hits


def _detect_deshihou_density(text: str) -> list[dict]:
    """G3：检测单段内「的时候」超过阈值。"""
    hits: list[dict] = []
    paragraphs = _split_paragraphs(text)
    for para in paragraphs:
        count = para.count("的时候")
        if count > _DESHIHOU_LIMIT:
            hits.append({
                "rule_id": "g3_deshihou",
                "pattern_name": f"本段「的时候」出现{count}次（上限{_DESHIHOU_LIMIT}）",
                "match_text": "的时候",
                "sentence": para[:200],
                "example_fix": "将「的时候」替换为「时」或重写句子为自然中文时序。",
                "quote_violation": True,
            })
    return hits


def _detect_metaphor_density(text: str) -> list[dict]:
    """G11：检测单段内「仿佛/似乎/好像」超过阈值。"""
    hits: list[dict] = []
    _re_metaphor = re.compile(r"仿佛|似乎|好像")
    paragraphs = _split_paragraphs(text)
    for para in paragraphs:
        count = len(_re_metaphor.findall(para))
        if count > _METAPHOR_LIMIT:
            hits.append({
                "rule_id": "g11_metaphor",
                "pattern_name": f"本段比喻词「仿佛/似乎/好像」出现{count}次（上限{_METAPHOR_LIMIT}）",
                "match_text": "仿佛/似乎/好像",
                "sentence": para[:200],
                "example_fix": "减少比喻词使用，用具体感官细节替代。「像……一样」的比喻应以叙事效率为先。",
                "quote_violation": True,
            })
    return hits


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
    if not isinstance(text, str) or not text:
        return []
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

    # ── 段落级扫描 ──
    for para_check in [_detect_short_sentence_chains, _detect_deshihou_density,
                        _detect_metaphor_density]:
        for pv in para_check(text):
            # 段落级规则同一段只报一次
            key = (pv["rule_id"], pv["sentence"][:80])
            if key in seen:
                continue
            seen.add(key)
            hits.append(pv)

    return hits


# ---------------------------------------------------------------------------
# AI 通顺性审查（G10 / G12 / G14 / 通顺性 / 口语化 / 翻译腔）
# 仅对含叙事锚点词的文本触发，对工具调用和纯代码内容不触发。
# ---------------------------------------------------------------------------

_NARRATIVE_ANCHORS = re.compile(
    r"(?:^|\b)(?:他|她|说：|码头|转身|站起|走到|坐下|门前|窗外|茶杯|卷宗"
    r"|书记|证据|码头|街|门|船|声|光|夜|夜色)\b"
)


def _has_narrative_anchors(text: str) -> bool:
    """检测文本是否含有足够多的叙事锚点词（>=5个）以判定为叙事文本。"""
    if not isinstance(text, str) or not text:
        return False
    return len(_NARRATIVE_ANCHORS.findall(text)) >= 5


_FLUENCY_SYSTEM_PROMPT = """\
你是中文叙事小说的风格审校编辑。用户将给你一段小说正文，请逐句通读并完成修正。

修正对象：叙事段落（非角色对话）中读起来卡顿、生涩、不通顺的句子，以及过度口语化或翻译腔的句法。

过度口语化包括但不限于：叙事段落中口语连词（就、也、还、都、然后、而且）堆砌，句尾“了”泛滥（不承载状态变化的语气词“了”），短促口语句式连续排列。你自行判断叙事段落中哪些口语化痕迹过重，需要改为文学性书面语。

翻译腔包括但不限于：句首“当……时”从句、形式主语“这使/这令/这让”、多层“的”嵌套、被动句在中性语境中滥用、“在……的情况下”结构、不必要的冠词（一个/一种）残留。你自行判断哪些句子的语序像英文直译而非自然中文，直接改写为地道中文。

不通顺包括但不限于：主语缺失、搭配不当、句法松散、单独拿出来中文母语者会卡住的句子。你自行判断并重写。

角色对话（引号内的文字）保持原样。具体情节、人物动作、场景描写的内容逻辑保持不变。只修改句法和用词，不添加或删除内容。

只输出修正后的完整正文，不要解释、不要标注修改位置、不要加任何前言或后记。"""


async def apply_fluency_fix(
    body: dict,
    call_upstream: Callable,
    result: dict,
) -> dict:
    """对含叙事锚点词的文本执行一次 AI 通顺性审查。

    无叙事锚点（工具调用、纯代码）时直接返回原 result。
    审查后若有修正，返回修正后的 result；无修正时返回原 result。
    """
    choices = result.get("choices", [])
    if not choices:
        return result
    content = choices[0].get("message", {}).get("content", "")
    if not content or not _has_narrative_anchors(content):
        return result

    # 构建一次性审查请求
    _prev_len = len(body["messages"])
    body["messages"].append({
        "role": "user",
        "content": f"{_FLUENCY_SYSTEM_PROMPT}\n\n---\n\n{content}",
    })
    try:
        corrected = await call_upstream()
        corrected_choices = corrected.get("choices", [])
        if not corrected_choices:
            return result
        corrected_content = corrected_choices[0].get("message", {}).get("content", "")
        # 空或过短的修正视为失败，回退
        if not corrected_content or len(corrected_content) < len(content) * 0.5:
            return result
        # 若内容完全相同（未修正），返回原 result
        if corrected_content == content:
            return result
        return corrected
    except Exception:
        logger.warning("apply_fluency_fix: 通顺性审查异常，回退到原响应", exc_info=True)
        return result
    finally:
        # 回滚追加的消息，恢复 body 到审查前状态
        body["messages"] = body["messages"][:_prev_len]


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
    call_alt_upstream: Callable | None = None,
) -> dict:
    """扫描 assistant 响应文本 → 反馈 → 重发上游 循环。

    每次重发时，将前一轮的 assistant 消息和风格反馈用户消息附加到
    body["messages"] 中，重新调用 call_upstream。

    当 call_alt_upstream 提供时：原 provider 重发耗尽仍存在违规，
    切换到异族模型重发一次；异族修正后仍违规则再切回原 provider 重发一次。

    返回最终的 response dict（无违规时直接返回原 result）。
    """
    if rules is None:
        rules = RULES

    # 构建 provider 队列：原生重试用 primary，交替时插入 alt
    _providers: list[tuple[str, Callable]] = [("primary", call_upstream)]
    if call_alt_upstream is not None:
        _providers = [("primary", call_upstream), ("alt", call_alt_upstream)]
    _active_fn: Callable = _providers[0][1]
    _provider_idx = 0  # 当前活跃 provider 的索引
    _best_result: dict | None = None
    _best_violations = float('inf')

    _total_retries = max_retries + 6 if call_alt_upstream is not None else max_retries
    for _retry in range(_total_retries):
        # 提取 assistant 文本内容
        choices = result.get("choices", [])
        if not choices:
            break
        content = choices[0].get("message", {}).get("content") or ""
        # 若 content 为空但存在 tool_calls，继续扫描 tool_call 参数
        if not content and not (choices[0].get("message") or {}).get("tool_calls"):
            break

        # 显式跳过标签：assistant 主动要求绕过风格扫描
        if _has_override_tag(content):
            choices[0]["message"]["content"] = _strip_override_tag(content)
            break

        violations = scan_violations(content, rules)

        # 扫描 tool_call 参数（Edit.new_string / Write.content）中的违规
        import json as _json
        _tc_args_violations: list[dict] = []
        _saved_tc = (choices[0].get("message") or {}).get("tool_calls")
        if _saved_tc:
            for _tc in _saved_tc:
                _fn = _tc.get("function") or {}
                _fn_args = _fn.get("arguments", "")
                if not _fn_args:
                    continue
                try:
                    _parsed = _json.loads(_fn_args)
                except (_json.JSONDecodeError, TypeError):
                    continue
                if _fn.get("name") == "Edit" and isinstance(_parsed.get("new_string"), str):
                    _tc_args_violations.extend(scan_violations(_parsed["new_string"], rules))
                elif _fn.get("name") == "Write" and isinstance(_parsed.get("content"), str):
                    _tc_args_violations.extend(scan_violations(_parsed["content"], rules))

        if not violations and not _tc_args_violations:
            break

        feedback = build_feedback_message(violations)
        # 将 tool_call 参数违规追加到反馈中
        if _tc_args_violations:
            _tc_feedback = build_feedback_message(_tc_args_violations)
            feedback = (
                feedback
                + "\n\n**注意：你即将通过 tool_call 写入文件的内容中也有违规，请同时修正 tool_call 参数**\n"
                + _tc_feedback
            )
        rid_list = sorted(set(v["rule_id"] for v in violations + _tc_args_violations))

        # 告警：控制台立即输出 + 持久化到专用日志
        ts = datetime.now().isoformat(timespec="seconds")
        header = f"[StyleGuard] {ts} R{_retry + 1}/4 violations={len(violations) + len(_tc_args_violations)} ids={rid_list}"
        lines = [header]
        for v in violations + _tc_args_violations:
            lines.append(f"  {v['rule_id']} {v['pattern_name']}: {v['sentence'][:80]}")
        alert = "\n".join(lines)
        with open(_ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(alert + "\n")

        logger.info("style_guard retry=%d violations=%d tc_args_violations=%d rule_ids=%s",
                     _retry + 1, len(violations), len(_tc_args_violations), rid_list)

        # 附加前一轮的 assistant 消息和反馈
        _msg = choices[0]["message"]
        # 若当前响应含 tool_calls，剥离 tool_calls 后再追加——避免 tool_calls
        # 消息缺少对应 tool_result 导致 DeepSeek "insufficient tool messages" 400 错误。
        # tool_calls 保留在 _saved_tc 中，修正完成后合并回最终结果。
        _tc_args_violated = bool(_tc_args_violations)
        if _saved_tc:
            logger.info(
                "style_guard retry=%d: 响应含 tool_calls，剥离后继续修正循环",
                _retry + 1,
            )
            _stripped_msg = dict(_msg)
            _stripped_msg["tool_calls"] = None
            body["messages"].append(_stripped_msg)
        else:
            body["messages"].append(_msg)
        body["messages"].append({
            "role": "user",
            "content": "请继续完成上文未完成的响应。下面检测到风格问题需要修正，同时参考下方的改进草稿，将修正融入你的下一次回复中。响应时不要重复反馈内容，不要以Changelog或修改列表的形式汇报你做了什么改动，直接输出修正后的全文。\n\n" + feedback,
        })

        result = await _active_fn()

        # 跟踪最优结果
        _scan_result = result.get("choices", [])
        if _scan_result:
            _scan_content = _scan_result[0].get("message", {}).get("content") or ""
            _scan_vns = scan_violations(_scan_content, rules)
            if len(_scan_vns) < _best_violations:
                _best_violations = len(_scan_vns)
                _best_result = result

        # 若重发后的响应含 tool_calls，回滚 body，保留修正文本
        # 并合并原始 tool_calls（避免工具调用丢失）
        # 但若原 tool_call 参数内含违规，则不合并，保留 LLM 重新生成的 tool_calls
        if (result.get("choices", [{}])[0].get("message", {}).get("tool_calls")):
            logger.warning(
                "style_guard retry=%d: 修正后响应含 tool_calls，保留文本并合并原始 tool_calls",
                _retry + 1,
            )
            del body["messages"][-2:]  # 回滚本轮追加的 assistant+feedback
            if _saved_tc and not _tc_args_violated:
                result["choices"][0]["message"]["tool_calls"] = _saved_tc
            # 切换 provider 继续修正
            _provider_idx = (_provider_idx + 1) % len(_providers)
            _active_fn = _providers[_provider_idx][1]
            continue

        # 本次重发后仍有违规：切换到下一个 provider 继续
        _provider_idx = (_provider_idx + 1) % len(_providers)
        _active_fn = _providers[_provider_idx][1]

    # 返回最优结果
    if _best_result is not None:
        result = _best_result

    return result
