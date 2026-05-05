"""廉价的提示词优化技术 + 内置 skills（in-process，无第二端口、0 额外 LLM 调用）。

拆分说明：
  - 文本常量 + 辅助函数 → skills_general.py
  - 消息转换（RE2 / CoT / readurls）→ skills_transform.py
  - `apply_cheap_optimizations`（编排入口）+ `extract_cot_output`（public）保留在此
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import httpx

from ..compatibility.deepseek_fixes import has_tools
from ..utils import append_to_system_message, find_system_message, prepend_to_system_message, sample_in_range

from .skills_general import (
    _SKILL_AVOID_AI_TICS,
    _SKILL_AVOID_FABRICATED_CITATIONS,
    _SKILL_COMPLEX_SENTENCE,
    _SKILL_COT_RESET,
    _SKILL_ASSUME_GOOD_INTENT,
    _SKILL_INDEPENDENT_ANALYSIS,
    _SKILL_INSTRUCTION_PRIORITY,
    _SKILL_JSON_MODE,
    _SKILL_NARRATOR_STANCE,
    _SKILL_NATURAL_TEMPERAMENT_CODING,
    _SKILL_PREFER_MULTIPLE_SOURCES,
    _SKILL_REASON_GENUINELY,
    _SKILL_SAFE_INLINED,
    _SKILL_SHOW_MATH_STEPS,
    _cot_eligible,
    _date_skill_line,
    _has_inlined_content,
    _is_json_mode,
)
from .skills_transform import (
    _apply_cot_reflection,
    _apply_readurls,
    _apply_re2,
    extract_cot_output,
)

logger = logging.getLogger(__name__)


async def apply_cheap_optimizations(
    body: Dict[str, Any],
    *,
    mode: Literal["coding", "creative"] = "coding",
    # A. 通用风格 skills (always active)
    avoid_negative_style: bool = True,
    natural_temperament: bool = True,
    contextual_register: bool = True,
    assume_good_intent: bool = True,
    instruction_priority: bool = True,
    independent_analysis: bool = True,
    reason_genuinely: bool = True,
    inject_date: bool = True,
    cot_reset: bool = True,
    # B. 求证 / 反幻觉 skills (model self-gates)
    show_math_steps: bool = True,
    prefer_multiple_sources: bool = True,
    avoid_fabricated_citations: bool = True,
    # C. 上下文相关 skills (narrow triggers)
    json_mode_hint: bool = True,
    safe_inlined_content: bool = True,
    # D. 消息转换 (mutates messages)
    re2: bool = True,
    cot_reflection: bool = True,
    readurls: bool = True,
    # 元功能：LLM-based system prompt 压缩（首次调一次模型，结果磁盘缓存复用）
    compressor: Optional[Any] = None,  # SystemPromptCompressor 实例；None 跳过压缩
    http_client: httpx.AsyncClient | None = None,
) -> Dict[str, Any]:
    """对请求体施加廉价的提示词优化（原地修改并返回 body）。

    mode 参数控制两条 skills 路径：
    - "coding"（默认）：使用 _SKILL_AVOID_NEGATIVE_STYLE_CODING +
      _SKILL_NATURAL_TEMPERAMENT_CODING（原版 skills，适合 code/math/技术分析）
    - "creative"：使用 _SKILL_AVOID_AI_TICS + _SKILL_NARRATOR_STANCE
      （不抑制情感表达，匹配通用创作要求）。B 组求证/反幻觉 skills 在
      creative 模式下跳过（math/citation/sources 对创作无益且增加合规心态）。

    分为四组：
    - A. 通用风格 skills（avoid_negative_style / assume_good_intent /
      natural_temperament / contextual_register / instruction_priority /
      independent_analysis / reason_genuinely / cot_reset / inject_date）
      （排序按语义连贯性：交互契约→输出风格→推理自主性→推理元认知）
    - B. 求证 / 反幻觉 skills（show_math_steps / prefer_multiple_sources /
      avoid_fabricated_citations）
    - C. 上下文相关 skills（json_mode_hint / safe_inlined_content）
    - D. 消息转换（re2 / cot_reflection / readurls）

    跳过条件：
    - 没有 messages
    - 含 tools / tool_choice（避免污染 function calling 提示词）

    CoT Reflection 额外条件：
    - 非流式（stream != True）
    - thinking 显式 disabled（V4 thinking=enabled 时自带 CoT，叠加无益）
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body
    if has_tools(body):
        return body
    # 防双重处理（同一 body 多次穿过）
    if body.get("_deepproxy_optimized"):
        return body
    body["_deepproxy_optimized"] = True

    # 1. readurls 最前：先把链接展开为内联文本，让后续 skills 能看到 [Content from ...]
    if readurls:
        await _apply_readurls(messages, client=http_client)

    # 2. 内置 skills（注入到 system prompt 前缀，按通用程度排序）
    skill_lines: List[str] = []

    # A. 通用风格（每请求激活）
    #
    # avoid_negative_style：coding + creative 共用 _SKILL_AVOID_AI_TICS（通用 AI 口癖禁止）
    # natural_temperament：按 mode 选择路径
    #   - coding：_SKILL_NATURAL_TEMPERAMENT_CODING（原版人格素描）
    #   - creative：_SKILL_NARRATOR_STANCE（叙事者立场，通用创作要求编码）
    # 其余 A 组 skills 两种模式共用
    is_creative = (mode == "creative")

    if avoid_negative_style:
        skill_lines.append(_SKILL_AVOID_AI_TICS)

    if assume_good_intent:
        skill_lines.append(_SKILL_ASSUME_GOOD_INTENT)

    if is_creative:
        if natural_temperament:
            skill_lines.append(_SKILL_NARRATOR_STANCE)
    else:
        if natural_temperament:
            skill_lines.append(_SKILL_NATURAL_TEMPERAMENT_CODING)

    if contextual_register:
        skill_lines.append(_SKILL_COMPLEX_SENTENCE)
    if instruction_priority:
        skill_lines.append(_SKILL_INSTRUCTION_PRIORITY)
    if independent_analysis:
        skill_lines.append(_SKILL_INDEPENDENT_ANALYSIS)
    if reason_genuinely:
        skill_lines.append(_SKILL_REASON_GENUINELY)
    # 注：inject_date 不进 skill_lines（也就不进 LLM 压缩缓存键），
    # 否则日期每天变化会让缓存每日全失效。改为在压缩后追加到 system 末尾。
    if cot_reset:
        skill_lines.append(_SKILL_COT_RESET)

    # B. 求证 / 反幻觉（模型自门控；对创作豁免）
    if not is_creative and show_math_steps:
        skill_lines.append(_SKILL_SHOW_MATH_STEPS)
    if not is_creative and prefer_multiple_sources:
        skill_lines.append(_SKILL_PREFER_MULTIPLE_SOURCES)
    if not is_creative and avoid_fabricated_citations:
        skill_lines.append(_SKILL_AVOID_FABRICATED_CITATIONS)

    # C. 上下文相关（仅窄触发条件下激活）
    if json_mode_hint and _is_json_mode(body):
        skill_lines.append(_SKILL_JSON_MODE)
    if safe_inlined_content and _has_inlined_content(messages):
        skill_lines.append(_SKILL_SAFE_INLINED)

    # 把 skills + 用户原 system 拼成完整 system prompt 后整体送 LLM 压缩
    skills_text = "\n\n".join(skill_lines) if skill_lines else ""
    sys_idx, user_sys_text, user_sys_compressible = find_system_message(messages)

    if skills_text and user_sys_text:
        combined = f"{skills_text}\n\n{user_sys_text}"
    elif skills_text:
        combined = skills_text
    elif user_sys_text:
        combined = user_sys_text
    else:
        combined = ""

    # 仅当 user system 是字符串时才走 LLM 压缩（多模态 list 跳过避免破坏结构）。
    if combined and compressor is not None and (not user_sys_text or user_sys_compressible):
        try:
            combined = await compressor.compress(combined)
        except Exception as e:
            logger.warning("system prompt 压缩调用失败，使用原文: %s", e)

    if combined:
        if sys_idx is not None and user_sys_compressible:
            messages[sys_idx]["content"] = combined
        elif sys_idx is not None:
            # 已有 system 但 content 是非字符串（多模态）—— 不动它，把 skills 插一条新的在前
            if skills_text:
                prepend_to_system_message(messages, skills_text)
        else:
            prepend_to_system_message(messages, combined)

    # 2.5 inject_date：在压缩之后追加到 system 末尾。
    # 日期每天变化，若进入压缩缓存键会让缓存每日全部失效；放在压缩外，
    # 同时位于 system 末尾确保最新日期始终对模型可见。
    if inject_date:
        append_to_system_message(messages, _date_skill_line(), dedup=True)

    # 3. RE2
    if re2:
        _apply_re2(messages)

    # 4. CoT Reflection
    if cot_reflection and _cot_eligible(body):
        _apply_cot_reflection(messages)
        body["_deepproxy_strip_cot"] = True

    return body
