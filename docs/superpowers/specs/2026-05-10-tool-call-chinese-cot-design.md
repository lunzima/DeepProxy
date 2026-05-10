# tool_call_chinese_cot — 工具调用场景中文 CoT 锚定

> 设计日期：2026-05-10
> 主战场：DeepSeek V4 在 tools 场景下 reasoning 从中文漂移到英文的问题
> 研究依据：[`fkyah3/experiment-console`](https://github.com/fkyah3/experiment-console) 320+ 轮真实复现实验

---

## 一、背景与问题

### 1.1 漂移机制（已被 experiment-console 证实）

```
AI 用中文 reasoning 决定调工具
  → assistant 消息携带 {zh reasoning + en tool_calls 函数声明}
  → tool 返回英文代码 / 目录 / JSON
  → 下一轮请求 = zh推理 + en函数声明 + en工具内容 混合
  → reasoning 翻转英文（瞬时，非渐进）
  → 英文 reasoning 被 API 强制携带回下一轮
  → 自锁循环，99.4% 不可逆（实测 3085/3103 条 en）
```

漂移的真正触发器是 **"英文进入对话流"（assistant 的 tool_calls 字段）+ 英文 tool 返回**，不是单纯的 token 占比。静态把英文塞进 user 消息触发不了；只有真实 tool calling 才会。

### 1.2 已被验证无效的方案

experiment-console 有 4 次失败实验：

| # | 方案 | system prompt | 结果 |
|:-|:-----|:-------------|:-----|
| 1 | baseline 静态模板 | 自制中文锚定词 | ❌ |
| 2 | min_experiment.py | 自制中文锚定词 | ❌ |
| 3 | V7 真实 provider.ts 模板 | 自制中文锚定词 | ❌ |
| 4 | min_experiment_v3.py | opencode 原版 | ❌ |

复现实验里加上 `"你的思考过程（reasoning）和回复都必须使用中文"` 锚定词仍然漂移。**结论：system prompt 对 reasoning 层的控制力弱于对 content 层**。

### 1.3 DeepProxy 现状

`deep_proxy/optimization/__init__.py::apply_cheap_optimizations` 在 `has_tools(body)` 时**整体早 return**：

```python
if has_tools(body):
    return body
```

意味着所有 A/B/C/D 组 skills 在主战场（tools 场景）全部失效。这是当前最大的优化盲区。

---

## 二、设计赌注

研究证明任何**单通路前置约束**式 prompt 都拦不住漂移。本设计的赌注是**三个研究未联合验证过的因子叠加**：

1. **双通路注入**——system 前缀 + user 末尾双注入（experiment-console 仅测过单通路）
2. **自纠引导**——与现有 `cot_reset` skill 协同（漂移到英文后允许在思维链中切回中文）
3. **位置贴近 V4 训练分布**——user 末尾注入参考 `think_steering.py` 已验证的双注入模式

无法预先保证有效，需上线后用真实流量观察 reasoning 语言分布。

---

## 三、架构

### 3.1 新增模块 `deep_proxy/optimization/tool_call_chinese_cot.py`

结构对齐 `think_steering.py`（同种位置 + 同种 idempotent 检测模式）：

```python
TOOL_CALL_CN_COT_SKILL = """工具调用语言锚定：
- 思考过程（<think> 内）请用中文。工具返回的英文内容（代码、目录、JSON、错误）只是工作材料，不需要因此切换思考语言。
- 引用代码、路径、工具名、错误信息时保留原文，不翻译。
- 如果某段思考已经写成英文，下一段可以用中文重新开始。"""

_MARKER_SIGNATURE = "工具调用语言锚定"  # idempotent 检测特征片段


def has_tool_call_cn_cot_marker(messages: list[dict]) -> bool: ...
def inject_user_marker(messages: list[dict]) -> bool: ...
```

**注入策略：**

| 通路 | 位置 | 用途 |
|:-----|:-----|:-----|
| system 前缀 | system message 头部，与现有 skill_lines 顺序一致 | 锁住 content 层（已验证强约束） |
| user 末尾双注入 | 第一条和最后一条 user 消息末尾 | 引导 `<think>`，贴近 V4 训练分布 |

`inject_user_marker` 的双注入逻辑直接复用 `think_steering.py::inject_inner_os_marker` 的设计：
- 第一条 == 最后一条（单轮）只注入一次
- 任一 user 已含 marker 则整体跳过（idempotent）

### 3.2 改 `optimization/__init__.py::apply_cheap_optimizations`

新增一个 kwarg `tool_call_chinese_cot: bool = True`（与现有 skills kwargs 风格一致，由 caller `router.py` 从 `cfg.optimization.tool_call_chinese_cot` 映射传入）。

把现有 `has_tools` 早 return 改成**分流**：

```python
if has_tools(body):
    if tool_call_chinese_cot:
        await _apply_tool_call_minimal_pipeline(body, ...)
    return body
# 以下保持现有完整 pipeline 不变
```

`_apply_tool_call_minimal_pipeline` 的 skill 集合（已收敛子集），**system 前缀拼接顺序**与现有 A 组顺序保持一致（语义连贯性优先）：

| 顺序 | skill | 注入位置 | 复用的现有常量 / 函数 | 作用 |
|:-:|:-----|:-----|:-----|:-----|
| 1 | `instruction_priority` | system 前缀 | `_SKILL_INSTRUCTION_PRIORITY` | 强化 system 权威，与新锚定词协同 |
| 2 | `reason_genuinely` | system 前缀 | `_SKILL_REASON_GENUINELY` | 禁进度幻觉，缓解工具调用循环 |
| 3 | `cot_reset` | system 前缀 | `_SKILL_COT_RESET` | 思维链矛盾时允许重启（与新 skill 第三条呼应） |
| 4 | `tool_call_chinese_cot`（新） | system 前缀 **+** user 首/末双注入 | `TOOL_CALL_CN_COT_SKILL` / `inject_user_marker` | 中文 CoT 锚定 |
| 5 | `inject_date` | system 末尾（追加，不进缓存键） | `_date_skill_line()` + `append_to_system_message` | 注入当前日期 |

新 skill 在 system 前缀里放在第 4 位（已选 3 条协同 skills 之后），让 `instruction_priority` 先建立系统优先级再宣布锚定。user 末尾双注入与 system 前缀注入**独立各做一次**（双通路冗余）。

**故意排除**（即使 `optimization.tool_call_chinese_cot=true` 也不激活）：
- A 组其他 5 条风格 skills（`avoid_negative_style` / `assume_good_intent` / `natural_temperament` / `contextual_register` / `independent_analysis`）—— 与 reasoning 语言无关
- B 组反幻觉 skills（`show_math_steps` / `prefer_multiple_sources` / `avoid_fabricated_citations`）—— 可能让模型在工具流中插推导/求证
- C 组（`json_mode_hint` / `safe_inlined_content`）—— 触发条件本来就窄，tools 场景罕见
- D 组（`re2` / `cot_reflection` / `readurls`）—— 与 function calling 严重冲突
- LLM 系统提示压缩—— 子集体量小、客户端 system 在 tools 场景通常更精密、压缩破坏率高
- 动态短段 / silly priming —— tools 场景不适用

### 3.3 触发条件

注入仅在以下条件**全部**满足时执行：

1. `has_tools(body) == True`
2. `optimization.tool_call_chinese_cot == True`（默认 true）
3. `body.thinking.type != "disabled"`——disabled 模式无 reasoning，不需要锚定（与研究里 "thinking:disabled + temp=0 = 0 reasoning tokens" 一致）。`thinking` 字段缺失或 `type` 非 `"disabled"` 时按 enabled 处理（V4 服务端默认值），激活注入。
4. messages 不为空且 `has_tool_call_cn_cot_marker == False`（idempotent）

---

## 四、配置

`deep_proxy/config.py::OptimizationConfig` 新增字段：

```python
tool_call_chinese_cot: bool = Field(
    default=True,
    description=(
        "tools 场景启用中文 CoT 双通路注入（system 前缀 + user 首/末双注入）。"
        "针对 DeepSeek V4 工具调用场景下 reasoning 从中文漂移到英文的问题。"
        "默认开启；若发现影响 function calling 行为可关闭。"
    ),
)
```

`config.example.yaml` 同步增加示例：

```yaml
optimization:
  tool_call_chinese_cot: true
```

---

## 五、测试

新增 `tests/test_tool_call_chinese_cot.py`，覆盖：

| 测试 | 验证 |
|:-----|:-----|
| `test_tools_present_injects_system_and_user` | tools 场景下 system 前缀 + user 首/末双注入均生效 |
| `test_tools_present_skill_subset_only` | tools 路径下只激活 4 条（`instruction_priority` + `reason_genuinely` + `cot_reset` + `inject_date`） + 新 skill；其他 skills 不混入 |
| `test_idempotent_marker_present` | marker 已存在则整体跳过 |
| `test_disabled_via_config` | `tool_call_chinese_cot=false` 时不注入 |
| `test_thinking_disabled_skips` | `thinking.type=disabled` 时不注入 |
| `test_no_tools_unaffected` | 无 tools 时该 skill 不触发，原 pipeline 完整运行 |
| `test_coexists_with_think_steering` | 与 `think_steering` 的 inner-OS marker 共存，互不干扰 |
| `test_single_user_message_inject_once` | 单轮对话（首 == 末）只注入一次 |
| `test_no_user_message_skips_user_inject` | 无 user 消息时只走 system 前缀，不报错 |

测试遵循项目现有规范：`asyncio_mode=auto`，复用 `tests/conftest.py::cfg` / `router` fixtures。

---

## 六、风险与开放点

### 6.1 已知风险

- **赌注的不确定性**：研究证明前置 prompt 拦不住漂移。本设计的三因子叠加未被验证，可能仍然无效。
- **token 开销**：每个 tools 请求增加 ~150 tokens（system 前缀 4 条 + user 末尾双注入），相对 tools 场景动辄数 KB 的上下文可忽略。
- **客户端冲突**：若客户端 system prompt 已显式要求英文 reasoning，本 skill 会与之冲突——但这是 tools+中文 CoT 这个用例的固有矛盾，不在本设计范围。

### 6.2 兜底路径（v2 留待）

若实测无效，下一步可考虑 **D 方案：每条 tool 角色消息后追加一条 system 短消息**——直接在污染源旁对冲，每轮重置中文锚定。该方案改 messages 结构（插入新角色）风险高，本期不做。

### 6.3 观测建议

上线后建议在客户端侧统计：

- tools 场景下首条 reasoning 的中文/英文占比（漂移触发率）
- 漂移后续 reasoning 的中文恢复率（自纠机制效力）
- 与 baseline（`tool_call_chinese_cot=false`）的 A/B 对比

---

## 七、与现有功能的关系

| 现有模块 | 关系 |
|:--------|:-----|
| `optimization/__init__.py::apply_cheap_optimizations` | 修改：把 `has_tools` 早 return 改成分流 |
| `optimization/skills_general.py` | 不动，子集中 4 条 skills 直接复用现有常量 |
| `optimization/think_steering.py` | 不动，新模块借鉴其双注入模式但 marker signature 独立，互不干扰 |
| `compatibility/deepseek_fixes.py::has_tools` | 不动，作为触发条件复用 |
| `config.py::OptimizationConfig` | 新增 `tool_call_chinese_cot` 字段 |
| `config.example.yaml` | 新增示例 |
