# 复杂度评分重设：跨编码 + CLI 写作的双向降级修复

## Context

`compute_complexity_score`（heuristic Layer 1）在长 agent 循环和"用 Claude Code 同时跑编码与文学创作任务"的混合场景下表现出三个方向的退化：

- **方向 A**：简单 user prompt → 几十轮 assistant tool 循环 → user-only scoring 看不到 emergent 复杂度 → 错误保留 flash → 推理能力不足。
- **方向 B**：复杂 user prompt 触发 Pro → 后续 "继续"/"对" 等简单 follow-up → user-only 评分骤降到简单档 → 即便对话上下文仍复杂 → 错误回到 flash。
- **方向 C**：复杂提问触发 Pro → 实际任务机械简单但需要多轮 → assistant 输出短小重复 → Pro 在 persist window 内无条件锁定直至自然过期 → 浪费推理预算。

期间发现重复代码与层级问题：`conversation_fingerprint` / `last_user_hash` / `flatten_messages` 等通用对话遍历工具住在 `optimization/flash_upgrade.py`，被 `cross_consult/` 和 `compatibility/reasoning_handler.py` 跨层引用。

我们曾尝试过若干"加维度"方案（assistant 侧加权扫描、agent_depth_score、Step 3.5 守卫、BERT 输入加 `[用户]/[助手]` 标记），但都属于对错误信号的补丁。本 spec 要求**回到第一性原理**：什么信号在编码任务和 CLI 创意写作任务上**同时**有效？

最终方案：以早期版本（commit `3c27c6f`）为基线，**保留 2 维**（keyword、math，user-only），**删除 2 维**（code、token + discount），**保留 1 维**（turn），**新增 2 维**（last_user_size、reasoning_density），并在 router 层加 Direction C hysteresis 主动降格。共 5 维评分 + 1 个降级机制。

---

## 设计

### 一、`compute_complexity_score` 重写（5 维 user-only / 推理密度）

```python
def compute_complexity_score(messages):
    if not messages:
        return ComplexityResult(0.0, "", 0)

    user_text  = flatten_messages(messages, user_only=True)
    user_turns = count_user_messages(messages)
    last_user  = last_user_text(messages)

    # 1. keyword（user-only，早期版本保留）— 用户语言意图信号
    keyword_hits  = sum(user_text.count(kw) for kw in _COMPLEXITY_KEYWORDS)
    keyword_score = min(keyword_hits * 0.30, 2.0)

    # 2. math（user-only，早期版本保留）— 数学/形式化任务强信号
    math_hits  = sum(1 for c in user_text if c in _MATH_SYMBOLS)
    math_score = min(math_hits * 0.50, 1.5)

    # 3. turn（user 消息数，早期版本保留）— 写作迭代 / 多轮 debug 持续度
    turn_score = min(user_turns / 3.0, 2.0)

    # 4. last_user_size（替换早期 token_score）— 当前"问题/请求"分量
    #    用最后一条 user 长度而非全量 token，避免 tool output / CLAUDE.md 膨胀噪声
    last_user_size_score = min(len(last_user) / 300.0, 3.0)

    # 5. reasoning_density（新增）— 最近 3 轮 assistant 的 reasoning_content
    #    平均长度。直接测量"模型在思考多努力"，跨编码/写作都有效；机械重复
    #    3 轮后信号降为零。滑动窗口 N=3（而非全历史平均）确保 Direction C
    #    在长 agent loop 中可响应——否则历史深度 reasoning 永久钉高均值。
    #    cap=8.0 = heuristic_threshold：重度 reasoning 单维度即可触发升格——
    #    这是 Direction A 唯一的触发路径（BERT user-only 看不到 grind）。
    _REASONING_WINDOW = 3
    asst = [m for m in messages if m.get("role") == "assistant"]
    if asst:
        recent = asst[-_REASONING_WINDOW:]
        reasoning_chars = sum(len(m.get("reasoning_content") or "") for m in recent)
        reasoning_score = min((reasoning_chars / len(recent)) / 500.0, 8.0)
    else:
        reasoning_score = 0.0

    score = keyword_score + math_score + turn_score + last_user_size_score + reasoning_score
    return ComplexityResult(round(min(score, 10.0), 2), user_text, user_turns)
```

加和上限 = 2.0 + 1.5 + 2.0 + 3.0 + 8.0 = 16.5 → clamp 10。reasoning cap 8.0 = `heuristic_threshold`，让重度 reasoning 单维度即可触发升格（Direction A 唯一触发路径）。

### 二、Direction C 主动降格机制

**`FlashUpgradeConfig.downgrade_threshold = 5.0`**（per_provider 可覆盖；MiMo 也设 5.0）。

校准依据（基于实测用户 agent loop 日志 18:26:12 请求的维度分解）：
- 实测 constants 基线（kw cap + math cap + turn + last_user_size 短 follow-up）= 4.56
- 阈值 3.0 时：reasoning windowed=0 后总分 4.56 仍远高于 3.0 → Direction C 从不触发
- 阈值 5.0 时：4.56 < 5.0 → 触发；最大 first-turn constants ceiling 6.83 仍守住升格

`router._maybe_upgrade` 在 Step 2（持久升格命中）之内**重新评估当前复杂度**：

```python
if self._upgrade_tracker.is_upgraded(messages, provider=provider_name):
    current_score = compute_complexity_score(messages).score
    downgrade_thr = cfg.threshold_for_provider(provider_name, "downgrade_threshold")
    if current_score < downgrade_thr:
        self._upgrade_tracker.clear(messages, provider=provider_name)
        logger.info("升格主动撤销: score=%.2f < downgrade=%.2f → %s", ...)
        # 不 return — 继续走 Step 3/4 让本轮按当下信号重新决策
    else:
        body["model"] = pro_model
        return
```

**Hysteresis 设计**：upgrade ≥ 8.0 (heur_thr) / 7.5 (mimo)、downgrade < 5.0 (downgrade_thr)，gap = 3.0 / 2.5 充分防振荡（reasoning windowed N=3 后单 chunk 不会跳变 ≥ gap）。

### 三、删除清单（今天 + 早期都需要删的部分）

| 删除项 | 原因 |
|--------|------|
| `code_score`（user 代码块计数） | 编码场景 Read 工具大量产生 ```，写作场景几乎不出现，不区分复杂度 |
| `token_score`（全文本 token 估算） | tool output / CLAUDE.md 灌爆 token 与下一轮复杂度弱相关，噪声大 |
| `discount` 机制 | 仅为修复 token_score 噪声而存在，token_score 删了不再需要 |
| `agent_depth_score`（今天加） | 单数轮数歧义大（机械 vs 推理同分），由 reasoning_score 取代 |
| `assistant_code_score`（今天加） | 已删 |
| 今天加的 kw/code/math 双源 scan | 回退到早期 user-only |
| Router Step 3.5 守卫（`agent_depth >= 5` 强升） | reasoning_score 覆盖 |
| BERT input 加 `[用户]/[助手]` 标记 + assistant tail | 模型在 user-only 上微调，输入分布漂移风险，回退 |
| `count_assistant_messages_since_last_user`（utils） | 无引用 |
| `flatten_messages(assistant_only=True)` flag | 无引用（`user_only` flag 保留） |
| `_last_assistant_text`（upgrade_router 局部） | 仅 BERT 用，BERT 回退后无需 |

### 四、保留清单

#### 早期版本继续保留
- `_COMPLEXITY_KEYWORDS` 完整列表（学术 + 文学性词项，覆盖两种工作）
- `_MATH_SYMBOLS`
- `flatten_messages(user_only=...)`
- `last_user_text` / `last_user_hash` / `count_user_messages` / `conversation_fingerprint`
- discount 机制中**不**直接保留（与 token_score 一起删）
- persist_turns / UpgradeTracker / RepeatUpgradeThrottle 全部基础设施

#### 今天工作保留（与 Direction C 相关）
- `FlashUpgradeConfig.downgrade_threshold` + per_provider 覆盖
- Router Step 2 hysteresis 重评估
- `config.yaml` MiMo `downgrade_threshold: 5.0`（与全局一致；hysteresis gap=2.5）
- Utilities 迁移到 `utils.py`（layering 修复）

### 五、关键文件修改

| 文件 | 修改 |
|------|------|
| `deep_proxy/optimization/flash_upgrade.py` | `compute_complexity_score` 重写为 5 维（kw + math + turn + last_user_size + reasoning_density）；删除 agent_depth_score / discount mechanism / today's 双源 scanning |
| `deep_proxy/optimization/upgrade_router.py` | `BertUpgradeRouter.score` 回退到 user-only 输入；删除 `_last_assistant_text` |
| `deep_proxy/router.py` | 保留 Step 2 hysteresis；删除 Step 3.5 + `count_assistant_messages_since_last_user` import |
| `deep_proxy/utils.py` | 删除 `count_assistant_messages_since_last_user`；`flatten_messages` 移除 `assistant_only` flag |
| `tests/test_agent_depth_scoring.py` | 重写为 `test_complexity_scoring_redesign.py`：测 5 维各自贡献 / 三方向覆盖 / Direction C hysteresis |
| `tests/test_flash_upgrade.py` | 现有 keyword/code/math 测试按 user-only 早期语义校准 |

### 六、三方向覆盖验证

| 方向 | 信号来源 | 触发路径 |
|------|---------|---------|
| **A** 简单 user + 复杂 grind | reasoning_score 单维度可达 8.0（cap = heuristic_threshold）；avg ~4000 字 reasoning/turn 即触发 | 总分跨过 `heuristic_threshold` (8.0 / MiMo 7.5)，**纯 heuristic 路径**（BERT user-only 看不到 grind） |
| **B** 复杂 user + 简单 follow-up "继续" | keyword_score 累积全部 user 历史 + reasoning_score 已累积；last_user_size 局部低但占比仅 3 / 16.5 | 总分稳在 downgrade_threshold (5.0) 之上 |
| **C** 升格后机械重复 | reasoning_density 下降到接近零（机械任务无推理） + 其它静态信号不动 | 中等复杂度初始 prompt（constants ≈ 4.5）总分跌破 5.0 → Step 2 hysteresis 主动撤销升格；max-complex 首轮（constants ~6.83）由于守住其下限继续保留 Pro |

---

## 验证

### 单元测试

```bash
python -m pytest tests/test_complexity_scoring_redesign.py tests/test_flash_upgrade.py -v
```

覆盖：
- 5 维各自贡献的边界（无 user → 0；纯 reasoning_content → 4.0 cap；纯 keyword → 2.0 cap 等）
- 三方向 fixture：每个方向构造典型 message 数组，断言 score 通过 / 未通过对应阈值
- Direction C hysteresis：tracker.set_remaining + 低分 messages → `_maybe_upgrade` 后 tracker 被 clear + model 回 flash
- 回归：早期 user-only kw/code/math 测试用例继续通过（删 code 维度的测试相应删除）

### 端到端验证

```bash
python -m deep_proxy.server
# 1. 简单一次性问答 → flash
# 2. 长 brief + 1 轮 assistant → flash 或 pro 由 reasoning_score 决定
# 3. 模拟 agent 多轮（手工 curl 提交带 reasoning_content 的历史）→ 升格
# 4. 模拟 Direction C：先升格再发 last user 简化 + 已 set tracker → 观察日志 "升格主动撤销"
```

---

## 范围之外

- 不重训 BERT — 输入回退到 user-only 后保持原训练分布
- 不引入新 LLM 调用做评分 — 保持启发式零额外上游调用约束
- 不动 persist_turns 计数语义（last_user_hash 变化才扣轮）— 长 agent loop 中"counter 不递减"是另一个已知限制，留待数据驱动后再处理
- 不增加 per_provider 阈值覆盖项 — 只在 MiMo `downgrade_threshold` 上有一处覆盖
