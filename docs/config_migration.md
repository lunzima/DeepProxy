# Config 迁移指南

DeepProxy 在 MiMo 集成 + Cross-Consult 引入后，配置文件由**单 provider 旧格式**升级为**多 provider 新格式**。`normalize_legacy_config` 提供加载期自动迁移，老 config.yaml 不动也能跑——但要用到 MiMo / cross_consult 必须显式迁移。

## 兼容矩阵

| config.yaml 形态 | 是否需手动改 | 行为 |
|---|---|---|
| **纯老格式**：只有 `deepseek:` + `coding_port` + `writing_port` | 否 | 加载期 `normalize_legacy_config` 注入 deepseek 单 provider，双端口都走 DeepSeek。MiMo / cross_consult 不可达。 |
| **纯新格式**：含 `providers:` + `ports:` | 否 | 直接生效；`deepseek:` / `coding_port` / `writing_port` 顶层字段被忽略。 |
| **半新格式**：只有 `providers:` 或只有 `ports:` 之一 | **是** | 启动期抛 `ValueError`——意图不明，必须补齐。 |
| **混合**：新格式 + 残留旧字段 | 否（但不推荐） | 新格式胜出，旧字段被静默忽略。建议清理掉以减少误读。 |

## 字段映射

| 旧字段 | 新字段 |
|---|---|
| `coding_port: 8000` | `ports: [{port: 8000, provider: deepseek, sampling: precise}]` |
| `writing_port: 8001` | `ports: [{port: 8001, provider: <name>, sampling: creative}]` |
| `deepseek.api_base` | `providers.deepseek.api_base` |
| `deepseek.api_key` | `providers.deepseek.api_key` |
| `deepseek.enable_reasoning` | `providers.deepseek.has_reasoning_content` |
| `deepseek.strip_unsupported_params` | （保留在 `deepseek:` 块，未迁移；只影响参数过滤兼容路径） |

`enable_reasoning` 改名为 `has_reasoning_content`，语义不变；`strip_unsupported_params` 暂未迁移至 Provider，老字段继续生效。

## 最小迁移示例

**老 config.yaml：**

```yaml
coding_port: 8000
writing_port: 8001
deepseek:
  api_base: https://api.deepseek.com
  api_key: sk-...
```

**等价新 config.yaml：**

```yaml
providers:
  deepseek:
    name: deepseek
    api_base: https://api.deepseek.com
    api_key: sk-...
    litellm_prefix: deepseek/
    flash_model: deepseek-v4-flash
    pro_model: deepseek-v4-pro
    has_reasoning_content: true
    has_thinking_param: true
    reasoning_effort_field: thinking.reasoning_effort
    reasoning_effort_value: max
    max_output_tokens: 384000
    context_window: 1000000

ports:
  - port: 8000
    provider: deepseek
    sampling: precise
  - port: 8001
    provider: deepseek
    sampling: creative
```

这次迁移**功能上无任何变化**——双端口仍然都走 DeepSeek，只是把 normalize 隐式注入的内容写到文件里。

## 启用 MiMo 第二 provider

在上面基础上追加 `mimo` provider 并把 writing 端口切到它：

```yaml
providers:
  deepseek:
    # ... 同上 ...
  mimo:
    name: mimo
    api_base: https://token-plan-cn.xiaomimimo.com/v1
    api_key: tp-...
    litellm_prefix: openai/
    flash_model: mimo-v2.5
    pro_model: mimo-v2.5-pro
    has_reasoning_content: true
    has_thinking_param: true
    reasoning_effort_field: reasoning_effort      # MiMo 是顶层字段
    reasoning_effort_value: high
    allowed_extra_params: [reasoning_effort, thinking]   # 让 LiteLLM 透传非 OpenAI 字段
    max_output_tokens: 128000
    context_window: 1000000

ports:
  - port: 8000
    provider: deepseek
    sampling: precise
  - port: 8001
    provider: mimo                       # ← writing 端口改成 mimo
    sampling: creative
```

关键差异：MiMo 走 `openai/` LiteLLM prefix（不是 `mimo/`，因为 LiteLLM 没有原生 mimo provider；MiMo API 是 OpenAI-compatible 的），并通过 `allowed_extra_params` 让 `reasoning_effort` / `thinking` 这两个非 OpenAI 标准字段经 `extra_body` 透传到上游。

## flash_upgrade per-provider 阈值

新格式支持每个 provider 独立阈值，覆盖全局默认：

```yaml
flash_upgrade:
  enabled: true
  router_type: bert
  bert_checkpoint: router_model
  router_threshold: 0.60               # 全局默认
  heuristic_threshold: 7.5
  persist_turns: 2
  per_provider:
    mimo:
      router_threshold: 0.65            # MiMo 起步保守，少升格到 pro
```

`per_provider` 没有的 key 走全局默认。BERT checkpoint 在所有 provider 间共享（同一模型做二分类）。

## 启用 Cross-Consult（可选）

虚拟工具，让 agent 在任意端口临时调用另一家族的 pro 模型：

```yaml
cross_consult:
  enabled: true
  tool_name: cross_consult              # 暴露给 agent 的工具名
  pairs:                                # 双向声明，无主副层级
    deepseek: mimo
    mimo: deepseek
  max_calls_per_request: 3              # 单次 client 请求内的 consult 次数上限
  # 超时旋钮统一由 streaming 段提供（cross_consult 与 plain 共用同一套）
  # consult 输入/输出不设武断上限：约束是 target provider 的 context_window / max_output_tokens
```

详细语义见 `docs/mimo_integration.md` §12。

## 常见错误

**`ValueError: config 含 providers 但缺少 ports`**
半新格式。补齐缺失的那一半即可。若只想跑老格式，把 `providers:` / `ports:` 都删掉，让 normalize 走兼容路径。

**MiMo 请求 400 `unknown field: reasoning_effort`**
`allowed_extra_params` 没配。LiteLLM OpenAI SDK 路径会校验 kwargs，非标准字段必须通过 `extra_body` 透传。把 `allowed_extra_params: [reasoning_effort, thinking]` 加到 `providers.mimo` 块。

**端口请求返回 DeepSeek 模型，但配置写的是 mimo**
检查 `ports:` 里该端口的 `provider:` 字段拼写。`provider:` 的值必须严格匹配 `providers:` 字典的 key。
