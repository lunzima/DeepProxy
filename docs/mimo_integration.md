# MiMo 集成 Spec

## 1. 背景与动机

`docs/agent_writing_qa_dilemma.md` 提出：DeepSeek V4 系列在中文创意写作场景下产生特征性口癖（物理测量式描写、声音物理化、归因句式），且同分布的判官（同样是 DeepSeek 系列）无法稳定识别本分布的失败模式——这是结构性同构困境。

引入异家族模型 Xiaomi MiMo v2.5 / v2.5-pro 作为创意写作 port 的独立上游，打破同分布闭环。MiMo 不充当 DeepSeek 的"校对者"，而是直接接管 `writing_port` (8001) 的全部流量；DeepSeek 继续服务 `coding_port` (8000) 的精确性任务。

之所以选 MiMo 而不是其他异家族模型：
- 中文原生，写作语感与目标场景匹配
- OpenAI / Anthropic 双端点兼容，接入成本低
- 与 DeepSeek 在协议层高度对齐（thinking 控制、reasoning_content、prefix cache、流式 SSE），改造面集中在路由层而非协议层
- 定价水平与 DeepSeek 相当，不引入成本结构性变化

## 2. 范围

**In scope**：
- 新增 provider 抽象，把上游 API 提到一等公民
- `writing_port` (8001) 流量切到 MiMo
- `mimo-v2.5` ↔ `mimo-v2.5-pro` 复用现有 `flash_upgrade` 升格路由
- per-port 的 `/v1/models` 列表
- 兼容层按 provider 差异化
- 向后兼容老 config 格式
- **Cross-Consult 虚拟工具**：任意 port（包括 coding）可临时调用异家族 pro 模型获取二次视角（见 §12）

**Out of scope**（明确不做）：
- MiMo 多模态（vision / audio / video）
- MiMo 联网搜索（与 readurls skill 重叠）
- MiMo TTS 系列模型
- MiMo `/anthropic/v1/messages` 原生端点直通（保留为未来优化点，写入代码注释）
- DeepSeek 路径的行为变更（cross_consult 工具注入除外）

## 3. Non-goals（架构反约束）

以下三条是硬约束，违反会被 review 打回：

### 3.1 定价数据独立，不共享常量
即使 MiMo 当前每档定价与 DeepSeek V4 数值一致，也**禁止**通过 import / 解构 / 默认值回退的方式共享常量。

```python
# 禁止
from .deepseek_pricing import PRICING
MIMO_PRICING = {**PRICING, ...}

# 禁止
def get_pricing(model: str) -> dict:
    return MIMO_PRICING.get(model, DEEPSEEK_PRICING[model])
```

正确做法：`mimo_pricing.py` 是 self-contained 模块，定价数据全量重写，与 `deepseek_pricing.py` 之间**没有 import 关系**。

理由：MiMo 已公告未来调价；两家当前数值相等是巧合而非协议。代码里不该编码这个巧合。

### 3.2 禁用"恩情"定型文
代码注释、docstring、log 消息、错误消息中**不得**出现以下模式：

- "致敬 / 借鉴 / 仿照 / 复刻 / 参考 DeepSeek"
- "向 DeepSeek 看齐 / 学习"
- "DeepSeek 的影响 / 启发"
- "雷军 / 小米 / DeepSeek" 的拟人化或情绪化措辞
- 任何将 provider 之间技术决策叙事化为致敬关系的表达

正确措辞示例：
- ❌ `# 这里仿照 DeepSeek 的做法处理 reasoning_content`
- ✅ `# reasoning_content 出现时按 OpenAI 扩展约定处理`
- ❌ `logger.info("MiMo 接入完成，向 DeepSeek 致敬")`
- ✅ `logger.info("provider=mimo bound to port=%d", port)`

理由：两个 provider 是技术等位关系，文档与代码记录技术事实即可，不需要叙事化对比。

### 3.3 不引入"primary / secondary"层级
provider 之间是**对等关系**，按 port 分流，不存在"主上游 + 备用上游"或"参考实现 + 派生实现"的层级。任何让人误以为有主次关系的命名（`primary_provider` / `fallback_provider` / `reference_provider`）都不要用。配置里用 provider 名 (`deepseek` / `mimo`) 直接指代。

Cross-Consult 工具（§12）尤其要遵守这一点：`pairs` map 双向定义，工具描述与系统提示词不出现主副措辞，不存在"默认 consult 方向"。

## 4. 架构变更

### 4.1 抽象层调整

当前：port 决定 sampling profile，上游隐式为 DeepSeek。
目标：port 决定 (provider, sampling)，上游显式查表。

```
client → :8000 → provider=deepseek + sampling=precise → DeepSeek API
client → :8001 → provider=mimo     + sampling=creative → MiMo API
                                    ↕  cross_consult  ↕
                            (任意 port 都可临时调对方 pro 模型)
```

### 4.2 模块拆分

```
deep_proxy/
  config.py                      # 加 Provider / Port / CrossConsult 配置模型 + 老格式兼容
  litellm_client.py              # call_* 接受 provider 参数；按 provider 选 api_base/key/prefix
  router.py                      # prepare_request 拿 port → 绑定 provider，往下传
  models_list.py                 # build_models_list(provider) → 按 provider 派发
  mimo_models.py                 # 真实模型 + 仿冒别名（self-contained）
  mimo_pricing.py                # USD/CNY 定价表（self-contained，见 3.1）
  cross_consult/
    __init__.py                  # 入口：注入工具、拦截 tool_use、执行 consult、重发
    executor.py                  # 单次 consult 调用的封装（context-free 上游调用）
    config.py                    # CrossConsultConfig pydantic 模型
  compatibility/
    base.py                      # 通用规范化（sanitize_stream_options 等）
    deepseek_fixes.py            # 仅 DeepSeek：legacy alias、reasoning_effort=max 注入
    mimo_fixes.py                # 仅 MiMo：reasoning_effort 顶层注入、payload 形状差异
    reasoning_handler.py         # 按 provider.has_reasoning_content 才生效
    error_mapper.py              # provider-agnostic
    anthropic_translator.py      # provider-agnostic
```

## 5. 配置 Schema

### 5.1 新格式

```yaml
providers:
  deepseek:
    api_base: "https://api.deepseek.com"
    api_key: "sk-..."
    litellm_prefix: "deepseek/"
    flash_model: "deepseek-v4-flash"
    pro_model: "deepseek-v4-pro"
    legacy_aliases:
      deepseek-chat:     {thinking: disabled}
      deepseek-reasoner: {thinking: enabled}
    has_reasoning_content: true
    has_thinking_param: true
    reasoning_effort_field: "thinking.reasoning_effort"   # 嵌套字段
    reasoning_effort_value: "max"
    thinking_disable_payload: {"thinking": {"type": "disabled"}}
    max_output_tokens: 384000
    context_window: 1000000

  mimo:
    api_base: "https://token-plan-cn.xiaomimimo.com/v1"
    api_key: "tp-..."
    litellm_prefix: "openai/"
    flash_model: "mimo-v2.5"
    pro_model: "mimo-v2.5-pro"
    legacy_aliases: {}
    has_reasoning_content: true
    has_thinking_param: true
    reasoning_effort_field: "reasoning_effort"            # 顶层字段
    reasoning_effort_value: "high"
    thinking_disable_payload: {"thinking": {"type": "disabled"}}
    max_output_tokens: 128000
    context_window: 1000000
    rate_limit_rpm: 100
    rate_limit_tpm: 10000000

ports:
  - port: 8000
    provider: deepseek
    sampling: precise
  - port: 8001
    provider: mimo
    sampling: creative

flash_upgrade:
  enabled: true
  router_type: bert
  bert_checkpoint: "router_model"
  router_threshold: 0.60
  heuristic_threshold: 7.5
  persist_turns: 2
  per_provider:
    mimo:
      router_threshold: 0.65          # 收紧起步，观察 MiMo pro 命中收益再调

cross_consult:
  enabled: true
  tool_name: "cross_consult"          # 工具暴露给 agent 的名字
  pairs:                              # 异家族配对（symmetric）
    deepseek: mimo
    mimo: deepseek
  max_calls_per_request: 3            # 单次 client 请求内的 consult 次数上限
  call_timeout_seconds: 30
  max_input_chars: 32000              # question + context 字符上限
  max_output_tokens: 4096             # consult 调用的输出 token 上限
  consult_system_prompt: |
    你被作为外部顾问从另一对话中召唤。直接回答问题，不寒暄，不复述问题。
    你没有该对话的上下文，只能依据本次提问中给出的信息作答。
    如果信息不足以回答，明确说明缺少什么。

optimization:
  compressor:
    provider: deepseek                # 压缩用 deepseek（成本/已验证）
```

### 5.2 向后兼容

`config.py` 加载阶段做规范化：

```python
def normalize_legacy_config(raw: dict) -> dict:
    if "providers" in raw and "ports" in raw:
        return raw
    # 老格式 → 新格式
    raw["providers"] = {"deepseek": {
        "api_base": raw.pop("api_base", "https://api.deepseek.com"),
        "api_key":  raw.pop("api_key"),
        # ... 其他字段从既有 ProxyConfig 字段映射
    }}
    raw["ports"] = [
        {"port": raw.pop("coding_port", 8000), "provider": "deepseek", "sampling": "precise"},
        {"port": raw.pop("writing_port", 8001), "provider": "deepseek", "sampling": "creative"},
    ]
    return raw
```

老 `config.yaml` 继续工作（双端口都打 DeepSeek，cross_consult 因无 pair 自动 no-op）；用户改 ports 表即可切换。

## 6. Provider 协议差异表

| 维度 | DeepSeek | MiMo |
|------|----------|------|
| `reasoning_effort` 位置 | `thinking.reasoning_effort`（嵌套） | `reasoning_effort`（顶层） |
| `reasoning_effort` 取值 | `max`（私有扩展） | `low` / `medium` / `high` |
| 关闭思考 | `{"thinking": {"type": "disabled"}}` | `{"thinking": {"type": "disabled"}}` |
| 默认思考行为 | 服务端默认 enabled | 文档说 disabled，token-plan-cn 端点实测 enabled（**显式设置不依赖默认**） |
| `reasoning_content` 字段 | 有 | 有 |
| Tools / tool_choice | OpenAI 标准 | OpenAI 标准 |
| 流式 SSE | OpenAI 标准 + `[DONE]` | OpenAI 标准 + `[DONE]` |
| Prefix cache | 有，`prompt_tokens_details.cached_tokens` | 有，同字段 |
| 上下文 | 1M | 1M |
| 最大输出 | 384K | 128K |
| `max_completion_tokens` | 接受 `max_tokens` | 接受 `max_tokens` 也接受 `max_completion_tokens` |
| Legacy alias | `deepseek-chat` / `deepseek-reasoner` 映射 | 无 |

## 7. 请求 Pipeline 改造

`router.py::prepare_request` 增加 provider 绑定步骤，其后所有步骤接受 provider 作为决策输入：

```
request → port → provider config bound
  1. legacy alias 应用（仅当 provider.legacy_aliases 非空）
  2. telemetry header 剥离（provider-agnostic，不变）
  3. reasoning_effort 注入：
       - DeepSeek: thinking.reasoning_effort = "max"
       - MiMo:     reasoning_effort = "high"（顶层）
     注入使用 provider.reasoning_effort_field 和 provider.reasoning_effort_value
  4. sampling 默认值（port → sampling profile）
  5. strip_unsupported_params（provider-agnostic 子集 + provider-specific 子集）
  6. ensure_reasoning_content_persistence（仅 provider.has_reasoning_content）
  7. sanitize_stream_options（不变）
  8. apply_cheap_optimizations（不变；skills 与 provider 无关）
  9. 动态短段注入（不变）
  10. 无厘头 expert priming（不变）
  11. flash_upgrade 路由（按 provider 选 flash/pro 模型名 + 阈值）
  12. cross_consult 工具注入（若 cross_consult.enabled 且当前 provider 有 pair；见 §12.4）
  13. litellm 调用（provider.litellm_prefix + provider.api_base + provider.api_key）
```

**关键决策**：写作 port 上无论客户端传什么 model 名（`claude-opus-4-7` / `gpt-4o` / `mimo-v2.5`），DeepProxy 一律映射到 `provider.flash_model`，再由 flash_upgrade 决定是否切 `pro_model`。沿用"port 决定一切"的现有约定。

响应路径上新增 cross_consult tool_use 拦截步骤（见 §12.5）。

## 8. flash_upgrade 复用与差异

### 8.1 共享部分（不改）
- BERT checkpoint (`router_model/`)：训练目标是任务难度二分类，provider-agnostic
- 启发式快速路径（token 数 + 复杂度关键词）：provider-agnostic
- 持续轮数 `persist_turns`：会话状态机，provider-agnostic
- 训练集 / 测试集：暂不重训

### 8.2 per-provider 部分

- **阈值**：`router_threshold` 和 `heuristic_threshold` 支持 `per_provider` 覆盖。MiMo 起步用 `router_threshold: 0.65`（比 DeepSeek 的 0.60 严），少升格，观察 pro 实际收益再调
- **模型名**：升格后切换的目标模型从 `provider.pro_model` 取（不再硬编码 `deepseek-v4-pro`）
- **session 状态隔离**：persist state 的 key 改为 `(client_session_id, provider_name)`，避免跨 provider 残留

### 8.3 观察期

跑两周收集 per-provider 指标：
- pro 命中率
- pro 升格后的用户主观反馈（是否值得贵 2.5x）
- BERT 分类器在 MiMo 场景的分布偏移（false positive / false negative 率）

若分布偏移显著，再决定是否需要 MiMo 专用训练集 + 单独 LoRA。

## 9. /v1/models per-port

每个 port 仅返回该 port 绑定 provider 的模型列表 + 仿冒别名。

```python
@app.get("/v1/models")
async def list_models(request: Request) -> dict:
    port = request.url.port
    provider = config.provider_for_port(port)
    return {"object": "list", "data": build_models_list(provider)}
```

`build_models_list` 派发：
```python
def build_models_list(provider: Provider) -> list[dict]:
    if provider.name == "deepseek":
        from .deepseek_models import build as build_deepseek
        return build_deepseek()
    elif provider.name == "mimo":
        from .mimo_models import build as build_mimo
        return build_mimo()
    raise ValueError(f"unknown provider: {provider.name}")
```

`mimo_models.py` 结构镜像 `deepseek_models.py`，但**内容独立**（见 3.1）：
- 真实模型：`mimo-v2.5`, `mimo-v2.5-pro`
- 仿冒别名：在 OpenAI / OpenRouter / Anthropic 命名空间下铺设别名（如 `claude-opus-4-7` → 内部路由到 `mimo-v2.5-pro`）
- 元数据：定价（从 `mimo_pricing.py`）、上下文长度（1M）、最大输出（128K）、display_name

## 10. 兼容层拆分

### 10.1 `compatibility/base.py`（新）
通用规范化：
- `sanitize_stream_options(req)` — 清空 `stream_options: {}` 这类无效空 dict
- `strip_common_unsupported(req)` — 移除所有 provider 都不支持的字段（`functions`、`user`）

### 10.2 `compatibility/deepseek_fixes.py`（瘦身）
保留：
- `normalize_legacy_model_name(req)` — `deepseek-chat` / `deepseek-reasoner` → `deepseek-v4-flash` + thinking 配置
- `inject_reasoning_effort_max(req)` — `thinking.reasoning_effort = "max"`

### 10.3 `compatibility/mimo_fixes.py`（新）
- `inject_top_level_reasoning_effort(req, value)` — 顶层 `reasoning_effort` 字段
- 暂无其他特定修复

### 10.4 `compatibility/reasoning_handler.py`（改）
入口加 `provider.has_reasoning_content` 守卫，false 直接 no-op。

## 11. Skills / Optimization 适配

| 模块 | 适配方式 |
|------|---------|
| skills_general / skills_transform | 不变（消息层改写，provider-agnostic） |
| dynamic_baskets | 不变 |
| silly_priming | 不变 |
| compressor | 配置项 `compressor.provider`（默认 deepseek），按配置调用对应 provider 做压缩 |
| think_steering（V4 `<think>` 引导） | 仅 `provider.has_thinking_param=true && provider.name=="deepseek"` 时启用 — MiMo 跳过（其 thinking 协议虽然兼容，但 V4 角色沉浸引导是基于 DeepSeek 训练分布的，对 MiMo 无意义且可能反效果） |
| tool_call_chinese_cot | 不变（中文 CoT 锚定与 provider 无关） |
| strip_telemetry | 不变（与 Anthropic 客户端有关，与上游无关） |

## 12. Cross-Consult 虚拟工具

### 12.1 目的

让**任意 port**（包括 coding）的请求能临时调用异家族的 pro 模型。打破单一分布的认知惯性，提供异分布二次视角，但不切换主调用链。

典型用法：
- 编码 port (DeepSeek) 上 agent 面对 API 命名 / UX 文案 / 注释风格 等创意子任务 → 询问 `mimo-v2.5-pro`
- 写作 port (MiMo) 上 agent 需要精确的逻辑校验 / 代码片段 / 算法解释 → 询问 `deepseek-v4-pro`
- 任一 port 上寻求异分布二次意见，打破当前模型的特征性盲点

### 12.2 工具规格

DeepProxy 在请求路径上向 `tools` 数组**注入**虚拟工具。客户端无需感知该工具的特殊性——它在 tools 数组中与其他工具地位等同。DeepProxy 在响应路径上**拦截**该工具的 tool_use，由 DeepProxy 自己执行，把结果以合成 tool_result 注入回会话，再向原 provider 重发。客户端最终只看到原 provider 的连续响应。

工具 schema：

```json
{
  "name": "cross_consult",
  "description": "向异分布模型家族请求对当前问题的视角。本工具由 DeepProxy 执行，不经过客户端文件系统或 shell。\n问题必须 self-contained——异家族模型没有本次会话的上下文。\n适用：跨领域子任务（写作里的逻辑、编码里的语感）、寻求二次验证、打破认知惯性。",
  "input_schema": {
    "type": "object",
    "required": ["question"],
    "properties": {
      "question": {
        "type": "string",
        "description": "明确的问题或子任务，self-contained"
      },
      "context": {
        "type": "string",
        "description": "可选背景。如果 question 引用了当前会话中的代码/文本，把片段放在这里"
      },
      "purpose": {
        "type": "string",
        "enum": ["second_opinion", "cross_domain_help", "style_check", "logic_check", "other"],
        "description": "调用意图，用于 telemetry，不影响行为"
      }
    }
  }
}
```

### 12.3 路由对偶

`cross_consult.pairs` 双向定义，对称无层级：

```yaml
pairs:
  deepseek: mimo
  mimo: deepseek
```

当前 port 绑定 provider A → 该 port 上的 cross_consult 调用 → 目标是 `pairs[A]` 的 `pro_model`。若 `pairs[A]` 未定义，cross_consult 在该 port 上**不注入**（agent 看不到工具）。

### 12.4 注入逻辑（请求路径，pipeline 第 12 步）

```
if cross_consult.enabled and current_provider.name in cross_consult.pairs:
    1. 在 request.tools 数组追加 cross_consult schema
       （若客户端原本没传 tools 数组，新建一个）
    2. 在 system prompt 末尾追加一段简短说明（见 §12.7）
    3. 在请求上打内部标记 _deepproxy_cross_consult_armed = true
```

### 12.5 拦截与执行（响应路径）

```
upstream response received
  ├─ 含 tool_use(name == cross_consult)?
  │   ├─ no  → 原样转发客户端
  │   └─ yes → 拦截，执行下述循环
  └─ 执行循环：
       1. 解析 tool_use 的 input：question / context / purpose
       2. 检查 per-request 调用计数 < max_calls_per_request；超限则合成错误 tool_result 并跳到步骤 5
       3. 调用 cross_consult.executor.consult(target_provider, question, context)：
            - 独立请求 target_provider.pro_model
            - system = cross_consult.consult_system_prompt
            - user = question + (context if any)
            - 不带 tools 数组（防递归）
            - 不走完整 pipeline（不注入 skills / baskets / cross_consult 自身）
            - 非流式 + max_tokens=cross_consult.max_output_tokens + timeout=call_timeout_seconds
            - 超时/错误：合成错误 tool_result
       4. 把 consult 结果封装成 tool_result block
       5. 把 (原 assistant 消息含 tool_use, 新 tool_result) 追加到 messages
       6. 用追加后的 messages 重发给原 provider（保持原 request 其他参数，stream 跟随客户端原意）
       7. 新响应：若再次含 cross_consult tool_use 且未超限，回到步骤 1；否则转发客户端
```

### 12.6 递归与限额

- consult 调用本身**不带 cross_consult 工具**——防止 A 调 B、B 又调 A 的链式递归
- consult 调用不注入任何 DeepProxy 优化（skills / baskets / priming / think_steering），保持"外部顾问"语义干净
- 单次 client 请求内 cross_consult 调用次数上限：`max_calls_per_request`（默认 3）
- 超限时 cross_consult 返回 tool_result 错误文本："cross_consult quota (N) exhausted for this request"，agent 自行处理
- consult 的输入字符数超 `max_input_chars` 时同样返回错误 tool_result，要求缩短

### 12.7 系统提示词增量

DeepProxy 注入工具时，向 system prompt 末尾追加：

```
[DeepProxy] 你可调用工具 `cross_consult` 向异家族模型请求第二视角。
适用：跨领域子任务、寻求二次验证、打破认知惯性。
注意：目标模型没有本次会话上下文，question 必须 self-contained，需要时把相关片段放在 context 字段。
本次会话最多调用 {max_calls_per_request} 次。
```

这段说明**不参与 prompt 压缩缓存**——`max_calls_per_request` 是用户配置项，且工具行为属于运行时元信息，混入缓存键会破坏 prefix cache 命中率。在压缩流程中显式跳过。

### 12.8 流式行为

v1：
- 非流式请求：直接走 §12.5
- 流式请求：检测到 cross_consult tool_use 收齐后，**停止向客户端流式输出**，buffer 剩余响应；执行 consult；重发原 provider 后**改为非流式**取得完整响应；以单个 chunk（或几个大 chunk）补完发给客户端

**已知缺陷**：流式体验在 consult 触发时降级为"前半流式 + 后半批量"。可接受——cross_consult 是低频工具，触发本身预期会延长本轮响应时间。v2 可考虑边 consult 边 keep-alive，或正确重启流。

### 12.9 成本与可观测性

- 每次 cross_consult 调用 = 1 次 target provider pro 调用 + 1 次原 provider 重发
- 日志结构化字段：`source_provider / target_provider / target_model / purpose / input_chars / output_tokens / duration_ms / status / call_index_in_request`
- 观察期内若 cross_consult 几乎不被调用，说明工具描述吸引力不足或场景过窄——调整描述或下调注入开销
- 若 cross_consult 被滥用（每请求都触满 3 次），调高门槛或降低 `max_calls_per_request`

### 12.10 与 Non-goals 的关系

cross_consult 实现必须遵守 §3.3：
- `pairs` map 双向定义，两个方向地位完全等同
- 配置项命名（`pairs` / `target` / `source` / `consult`）不出现 primary / secondary / fallback / reference
- 工具描述、系统提示词增量、日志消息中不出现"主模型 / 副模型 / 备用模型"措辞
- 实现层不存在"默认 consult 方向"——必须由当前 port 的 pair 显式查得

### 12.11 双家族 Awareness + 标签触发的整轮重定向（v0.4）

§12.1–§12.10 的 cross_consult 工具机制让 agent 能借一次外援。§12.11 在此之上补两块能力，目的是把"本代理桥接两个异家族 LLM"这件事变成 agent **能感知、可主动选用**的架构事实，而不只是隐式的后台能力。

#### 互补性

| 维度 | `cross_consult` 工具调用（§12.1–§12.10） | 标签触发的整轮重定向（§12.11） |
|------|------|------|
| 触发方 | agent 主动 emit tool_call | 上游/agent 在 user 消息中插入字面标签 `[本轮对话使用不同家族的大语言模型]` |
| 范围 | 单次咨询，原对话继续在源 provider | 当前轮 + persist 窗口内有限若干轮走目标 provider，窗口耗尽后自动回源 |
| 目标模型层级 | 始终强制 pro 模型 | 走目标 provider 自身的 flash_upgrade 分级路由 |
| 上下文 | 目标模型不见会话历史（question self-contained） | 目标模型接管整轮完整对话历史 |
| 适用 | "借一次外援"、独立子问题、第二意见 | "多轮次换家"、任务本质更适合对方家族 |

#### Awareness（system prompt 状态披露）

`cross_consult.awareness_enabled=true`（默认）时，`inject_into_request` 在原有 tool addendum 之前**前置**一段约 10–14 行中文披露，告诉 agent：

1. 本代理桥接哪两个家族（source + target，按当前已应用任何重定向后的"当前家族"视角呈现）
2. 三条可选路径——保持/调用 cross_consult/插入标签
3. 标签的字面形式必须严格使用

awareness 文案刻意**不暴露**以下细节，避免文案随实现演进失效或被 agent 针对性 prompt engineering：
- BERT / router / heuristic / threshold / score / persist_turns / 具体阈值数值
- 目标家族会用 flash 还是 pro 档（只说"按任务自适应"）

#### 标签重定向

`cross_consult.redirect_enabled=true`（默认）时，FastAPI 端点在调用 `prepare_request` 之前调 `cross_consult.redirect.resolve_redirect`，它：

1. 用 `compiled_redirect_pattern()` 扫描 `body["messages"]` 中的 user 消息
2. **最后一条 user 消息**含标签 → 触发重定向，置 `RedirectTracker` 计数为 `redirect_persist_turns + 1`（含本轮）
3. 所有 user 消息中的标签都会被剥离（历史消息中的标签也清理掉，避免上游模型在 history 看到 meta-instruction 被误导），但触发判定只看最后一条
4. 未命中标签 → 询问 `RedirectTracker.is_redirected()`，仍在 persist 窗口内同样返回目标 provider
5. 命中 → 把 source_provider 替换为目标 provider，下游 `prepare_request` 全部步骤按目标 provider 走（包括 flash_upgrade 阈值、reasoning_effort 注入位置、cross_consult tool 注入用对端做 pair 查找）
6. `pairs` 中查不到对偶 / 目标 provider 不在 `config.providers` → log warning 并 fail-open（返回 None，不重定向）

`RedirectTracker` 模式仿 `optimization.flash_upgrade.UpgradeTracker`：in-memory `OrderedDict` LRU 上限 512，key = `(conversation_fingerprint, last_user_hash, source_provider_name)`，复用 `flash_upgrade` 的 fingerprint / last_user_hash 计算（避免重复实现）。重定向状态与升格状态互不干扰、独立维护。

`sampling_profile` **不**随重定向变化——标签是"换 provider"而非"换写作风格"，入站 port 的 profile 含义对用户更稳定。

#### MiMo per-provider 阈值反向（配套调整）

`config.yaml` 中 MiMo 的 `flash_upgrade.per_provider` 覆盖方向反转：

```yaml
per_provider:
  mimo:
    router_threshold: 0.60      # 从 0.70 降到 0.60（全局 0.65）
    heuristic_threshold: 7.5    # 新增（全局 8.0）
```

原意"MiMo 起步保守、少升格"与实际能力分布不符：MiMo-v2.5 flash 略弱于 V4-flash，温和偏向升格更合理。值取小幅 nudge 而非大跨度，避免 mimo-pro 调用成本失控。重定向到 MiMo 时这套阈值生效——即"目标家族自动适配层级"的具体兑现。

#### 与 cross_consult 工具的交互

- 重定向后的 provider 视角下，`cross_consult` tool 仍由 `inject_into_request` 注入：以重定向后的 provider 名做 `pair_for` 查找，意味着重定向到 MiMo 的对话里工具能回查 DeepSeek，对称工作
- 重定向决策在 `prepare_request` 之前发生，所以 prepare_request 的 sentinel 防递归路径（`_deepproxy_cross_consult_internal`）天然也阻断重定向（resolve_redirect 早期返回 None）

## 13. 测试

`tests/` 新增：
- `test_provider_routing.py` — port → provider 绑定正确性
- `test_mimo_pipeline.py` — MiMo 路径的 reasoning_effort 注入位置 / 取值 / payload 形状
- `test_mimo_upgrade.py` — flash_upgrade 在 MiMo 上的阈值生效与模型切换
- `test_mimo_models_list.py` — `/v1/models` 在 MiMo port 上的响应
- `test_legacy_config_compat.py` — 老 config.yaml 仍能启动并双端口走 DeepSeek
- `test_cross_consult_injection.py` — 工具注入到 tools 数组 + system prompt 增量 + pairs 缺失时不注入
- `test_cross_consult_execution.py` — tool_use 拦截、consult 调用、tool_result 合成、重发原 provider；用 mock provider
- `test_cross_consult_limits.py` — max_calls_per_request、max_input_chars、timeout 三种限额
- `test_cross_consult_recursion_guard.py` — consult 内部请求不携带 cross_consult 工具
- `test_cross_consult_symmetry.py` — deepseek→mimo 和 mimo→deepseek 行为对称
- `test_cross_consult_redirect.py` — §12.11 标签命中 / 历史标签剥离 / persist 窗口衰减 / fail-open / 哨兵防递归
- `test_cross_consult_awareness.py` — §12.11 awareness 段含两 provider 名 + 三选项 + 字面标签，不泄露 BERT/threshold/档位等内部细节

`tests/integration/`（需 MiMo key + DeepSeek key）：
- `test_mimo_smoke.py` — 一次非流式 + 一次流式 + 一次 tools 调用真实打通
- `test_cross_consult_smoke.py` — 真实跨家族调用，验证 round-trip 正确

## 14. 实施顺序

1. **provider 抽象层**：config.py 新增 Provider/Port 模型 + 老格式 normalize；不接业务逻辑
2. **router/litellm_client 接 provider 参数**：所有调用点显式传 provider；DeepSeek 默认路径保持完全不变
3. **compatibility 拆分**：base / deepseek_fixes / mimo_fixes / reasoning_handler 守卫
4. **MiMo 端到端走通**：mimo_fixes 写完 → writing_port 配置切到 mimo → integration test 通过
5. **flash_upgrade 接 MiMo**：per_provider 阈值 + pro_model 取自 provider 配置 + session key 加 provider
6. **/v1/models per-port + mimo_models + mimo_pricing**
7. **Cross-Consult 工具**：cross_consult 模块（executor + 注入逻辑 + 响应拦截 + 重发循环 + 限额）；先非流式跑通，流式按 §12.8 v1 方案接入
8. **跑全套单元测试 + integration smoke**
9. **观察期**（2 周）：收集 pro 命中率、cross_consult 触发率与命中分布、用户主观反馈、阈值合理性

## 15. 待运行时验证项

实施过程中需用真实流量验证：

- token-plan-cn 端点的默认 `thinking` 行为是否在所有模型上一致（探针只测了 mimo-v2.5）
- MiMo 流式 chunk 中 `reasoning_content` 与 `content` 是否会同 chunk 出现（探针只见到分开 chunk）
- MiMo `tool_calls` 流式增量格式（探针只测了非流式 tool_calls）
- MiMo 在长上下文（>256K）下的实际延迟与稳定性
- MiMo 错误码格式与 `error_mapper` 现有映射的兼容性（探针只见到 400 / 401）
- cross_consult 在不同 client（Claude Code / OpenAI SDK 等）下，注入工具的兼容性——尤其是工具数组本来为空 vs 已有工具时的合并行为

这些不阻塞实施，但任一项失败时要在对应模块加适配。

## 16. 文档更新

实施完成后：
- `CLAUDE.md`：更新架构概览图、双端口说明、配置示例、cross_consult 章节
- `config.example.yaml`：补完 providers + ports + cross_consult 段
- `README.md`：如果有提到上游模型或工具，更新
- `docs/agent_writing_qa_dilemma.md`：在尾部追加"DeepProxy 的应对：写作 port 已切 MiMo + cross_consult 提供异分布二次视角"，链接本 spec
