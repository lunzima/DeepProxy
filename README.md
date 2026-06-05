# DeepProxy

> **提升 DeepSeek 官方 API 兼容性的代理服务器**

基于 [LiteLLM](https://github.com/BerriAI/litellm) 构建，提供一个完全 OpenAI 兼容的端点，解决 DeepSeek API 的兼容性问题，并集成 in-process 提示词优化技巧。

---

## 架构

```
客户端 (OpenAI SDK / Anthropic SDK) → DeepProxy (:8000 / :8001)
  ├─ [兼容层] 参数过滤 / 老模型别名 / reasoning / 错误映射 / Anthropic↔OpenAI 翻译
  ├─ [模型层] 三生态 /v1/models（OpenAI/OpenRouter/Anthropic 同条目共存：定价 / 上下文长度 / display_name / 仿冒别名）
  ├─ [升格层] Flash→Pro 选择路由器（BERT 二分类 + 启发式快速路径，per-provider 阈值 + per-port 动态阈值闭环）
  ├─ [优化层] 内建 skills（A/B/C/D 四组，0 额外 LLM 调用）
  │            + LLM 压缩（首次调一次，结果磁盘缓存复用）
  │            + 动态短段注入（场景化中文短段提示词）
  ├─ [协作层] Cross-Consult 虚拟工具（可选）：临时调用异家族 provider 的 pro 模型取第二视角
  └─ [路由层] LiteLLM ──┬──→ DeepSeek API (api.deepseek.com)        # :8000 coding/precise
                        └──→ MiMo API (token-plan-cn.xiaomimimo.com) # :8001 writing/creative
```

## 解决的问题

| 问题 | 说明 |
|------|------|
| **参数兼容** | 自动过滤 DeepSeek 不支持的旧 OpenAI 参数（`functions` / `user`），避免 400 |
| **V4 别名层** | `deepseek-chat` / `deepseek-reasoner` 自动映射到 `deepseek-v4-flash` 并隐含正确的 `thinking.type` |
| **Reasoning 处理** | 保留 `reasoning_content`，多轮缓存自愈；模型剥离时从原始对象兜底恢复 |
| **错误映射** | 将 DeepSeek/LiteLLM 错误转换为标准 OpenAI 格式；429/5xx 指数退避重试 |
| **提示词优化** | 内建 15+ 廉价 skills（通用风格 / 反幻觉 / 上下文 / 消息转换），全 in-process，0 额外 LLM 调用 |
| **Flash→Pro 升格** | 四层路由器自动评估请求复杂度，高复杂度请求升格到 Pro（BERT 二分类 + 启发式快速路径，per-provider 阈值 + per-port 动态阈值闭环） |
| **多 provider 路由** | 每个端口绑定一个 provider（coding→DeepSeek，writing→MiMo）；writing 端口可选配加权 `model_pool` 逐请求跨家族随机选模型 |
| **Cross-Consult** | 可选虚拟工具，让 agent 临时调用异家族 provider 的 pro 模型获取第二视角；另含 user 标签触发的整轮 provider 重定向 |
| **Anthropic 兼容** | 将 Anthropic Messages API 请求转换为 OpenAI 格式路由到上游，支持流式和非流式 |
| **模型列表** | 三生态 `/v1/models`：单条目同时含 OpenAI / OpenRouter（定价/上下文）/ Anthropic（display_name + 社区扩展 max_input_tokens/max_tokens）字段，响应顶层带 Anthropic 分页（first_id/last_id/has_more）。故意不输出 `capabilities`（避免谎报代理未实现的 context-management beta） |
| **克隆模型** | 将 pro/opus/codex 等仿冒模型别名映射到对应的上游实际模型 |

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 配置

复制配置模板并填入 DeepSeek API key：

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，将 deepseek.api_key 设为你的 key
```

或通过环境变量传入（优先级高于 config.yaml）：

```bash
set DEEPSEEK_API_KEY=sk-your-deepseek-api-key
```

### 3. 启动

```bash
python -m deep_proxy.server
```

默认绑定两个端口：
- **Coding 端口** `http://0.0.0.0:8000`  → 精确采样（code/math/逻辑）
- **Writing 端口** `http://0.0.0.0:8001`  → 创作采样（RP/创意写作/通用聊天）

### 4. 使用

任何支持 OpenAI SDK 的工具只需修改 `base_url`：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-proxy-key",    # 可选，如配置了 api_key
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

通过 curl：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 5. 健康检查

```bash
curl http://localhost:8000/health
```

## 配置说明

完整配置项见 [`config.example.yaml`](config.example.yaml)（复制为 `config.yaml` 并填入 API key）；老（v0.1.x）格式 → 新格式迁移见 [`docs/config_migration.md`](docs/config_migration.md)。关键结构（providers + ports 新格式）：

```yaml
host: "0.0.0.0"

# 多 provider 路由：每个 provider 一段
providers:
  deepseek:
    api_base: https://api.deepseek.com
    api_key: ""                        # 填入你的 DeepSeek API 密钥
    flash_model: deepseek-v4-flash
    pro_model: deepseek-v4-pro
  # mimo: ...                          # 可选第二 provider，见 config.example.yaml

# 端口绑定：每个 port 选一个 provider + 一个采样 profile
ports:
  - { port: 8000, provider: deepseek, sampling: precise }   # → precise_sampling
  - { port: 8001, provider: deepseek, sampling: creative }  # → creative_sampling

optimization:
  enabled: true
  compress_skills: true                # LLM 压缩 + 磁盘缓存
  dynamic_baskets: true                # 场景化中文短段注入
  # ... 完整 skills 开关见 config.example.yaml

creative_sampling:                     # RP / 创作 / 通用写作
  temperature_min: 0.90
  temperature_max: 1.20
  top_p_min: 0.90
  top_p_max: 0.97
  # presence_penalty / frequency_penalty 略

precise_sampling:                      # 编程 / 数学 / 逻辑
  temperature_min: 0.25
  temperature_max: 0.45
  top_p_min: 0.95
  top_p_max: 0.95
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | - |
| `DEEPSEEK_API_BASE` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `PROXY_HOST` | 监听地址 | `0.0.0.0` |
| `PROXY_CODING_PORT` | Coding 端口 | `8000` |
| `PROXY_WRITING_PORT` | Writing 端口 | `8001` |
| `PROXY_API_KEY` | 代理认证密钥(可选) | - |
| `OPTIMIZATION_ENABLED` | 启用提示词优化 | `true` |
| `LOG_LEVEL` | 日志级别 | `info` |
| `DEEPPROXY_RELOAD` | 热重载模式（仅首个端口生效，uvicorn 限制） | `false` |

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全 (OpenAI 完全兼容) |
| `/v1/messages` | POST | Anthropic Messages API 兼容（请求被转换为 OpenAI 格式后路由到 DeepSeek） |
| `/v1/models` | GET | 列出可用模型（三生态：OpenAI / OpenRouter / Anthropic 字段共存，含定价/上下文长度/display_name/仿冒别名 + Anthropic 分页字段） |
| `/health` | GET | 健康检查 |

> 注：FIM (`/v1/completions`) 已下线——DeepSeek 官方 FIM 端点不支持 reasoning，需 FIM 的客户端请直连 DeepSeek `/beta/completions`。

## Windows 一键接入 Claude Code

DeepProxy 暴露 Anthropic 兼容的 `/v1/messages` 端点。在 Windows 上一键配置 Claude Code 所有相关环境变量（用户级永久、无需管理员）：

1. 启动 DeepProxy（双击 `start.bat`）
2. 双击 `tools\setup_claude_code_env.bat`（或在终端：`tools\setup_claude_code_env.bat`）
3. 打开**新终端**运行 `claude`

脚本读取 `config.yaml` 的 `host` / `coding_port` / `api_key`，写入以下用户级永久环境变量：

| 变量 | 默认值 |
|------|--------|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:<coding_port>` |
| `ANTHROPIC_AUTH_TOKEN` | `config.yaml` 的 `api_key`（缺省 `dummy`） |
| `ANTHROPIC_MODEL` | `deepseek-v4-pro[1m]` |
| `ANTHROPIC_SMALL_FAST_MODEL` | `deepseek-v4-flash` |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `false`（从源头关掉 billing header；代理层也独立剥离作为兜底） |

同时主动删除 `ANTHROPIC_API_KEY` 以避免与 `AUTH_TOKEN` 优先级冲突（已设置时会先确认）。

**参数**：

- `-DryRun`：只打印不写
- `-Uninstall`：删除上述 5 个变量
- `-Writing`：指向 `writing_port`（默认 coding）
- `-Force`：跳过 `ANTHROPIC_API_KEY` 覆盖确认（CI 场景）

示例：

```cmd
tools\setup_claude_code_env.bat -DryRun
tools\setup_claude_code_env.bat -Uninstall
```

代理层另独立剥离 `^x-anthropic-[a-z-]+:.*$` 形式的伪 header 行（Claude Code 2.1.42+ 的 billing header 含 session hash，会破坏 prefix cache），由 `optimization.strip_client_telemetry` 控制，默认开启。即便客户端未设置 `CLAUDE_CODE_ATTRIBUTION_HEADER=false`，代理也能保证下游接收稳定的前缀。

## 提示词优化（Skills Pipeline）

所有优化在请求管道内顺次执行，全 in-process，无额外 LLM 调用（压缩器除外）。

**按通用程度分四组：**

### A. 通用风格 skills（每请求激活）
- `avoid_negative_style` — 禁说教套话与情感抚慰套话
- `assume_good_intent` — 合理意图假设
- `natural_temperament` — 内在气质 priming
- `contextual_register` — 句法复杂度匹配内容密度
- `instruction_priority` — system 最高权威
- `independent_analysis` — 自主推理（反谄媚）
- `reason_genuinely` — 真实推理，禁进度/时间幻觉
- `cot_reset` — 推理出现严重矛盾时允许在思维链中显式重启
- `tool_call_chinese_cot` — tools 场景中文 CoT 双通路锚定
- `inject_date` — 注入当前 UTC 日期

### B. 求证 / 反幻觉 skills（模型自门控）
- `show_math_steps` — 闭合式数学展示推导
- `prefer_multiple_sources` — 争议性事实多来源权衡
- `avoid_fabricated_citations` — 不编造 URL/论文/DOI

### C. 上下文相关 skills（窄触发）
- `json_mode_hint` — json_object 时注入 JSON-only 指令
- `safe_inlined_content` — readurls 内容视为 DATA

### D. 消息转换 skills（实验性，默认关闭，需显式启用）
- `re2` — 复制最后一条 user 消息
- `cot_reflection` — 非流式 + thinking=disabled 时 `<thinking>/<output>` 引导
- `readurls` — 检测 URL 并内联网页正文

### LLM 压缩器（元功能）
首次请求时，将所有 skills + 用户 system prompt 合并，调一次 LLM 压缩到最短同义版，按 `sha256(version + model + text)` 持久化到磁盘缓存文件。后续相同配置的请求直接命中缓存，0 上游调用。`inject_date` **不**进压缩缓存键（在压缩之后才追加到 system 末尾），故缓存跨天持久、不每日刷新。

## 项目结构

```
deep_proxy/
├── deep_proxy/
│   ├── main.py / server.py / router.py   # 端点 / 启动 / 核心路由
│   ├── config.py / providers.py / pool.py # 配置模型 / Provider 绑定 / 加权模型桶
│   ├── litellm_client.py                  # LiteLLM 调用封装（流式/非流式）
│   ├── models_list.py + *_models.py + *_pricing.py  # /v1/models 三生态构建器 + 定价
│   ├── compatibility/                     # 参数过滤 / 别名 / reasoning / Anthropic 翻译 / MiMo 修复
│   ├── cross_consult/                     # 虚拟工具 + 双家族重定向 + 流式协作
│   └── optimization/                      # skills / 压缩器 / 动态短段 / flash_upgrade / 动态阈值
├── router_model/                # 微调后的 BERT 路由器（中文 BERT-small + LoRA）
├── datasets/                    # 训练 / 测试数据
├── tools/                       # 开发工具（BERT 训练脚本 + Claude Code 环境配置）
├── tests/                       # pytest 套件（默认排除 tests/integration）
├── config.yaml / config.example.yaml      # 默认配置 / 配置模板
├── start.bat                    # Windows 启动脚本
├── QWEN.md / CLAUDE.md          # 开发上下文指南（CLAUDE.md 为符号链接，含完整模块清单）
└── README.md / LICENSE
```

> 完整模块级结构与请求管道说明见 [`CLAUDE.md`](CLAUDE.md)。

## 开发

```bash
# 热重载模式（代码修改后自动重启）
set DEEPPROXY_RELOAD=true
python -m deep_proxy.server
```

## 许可

MIT
