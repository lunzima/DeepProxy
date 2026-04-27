# DeepProxy

> **提升 DeepSeek 官方 API 兼容性的代理服务器**

基于 [LiteLLM](https://github.com/BerriAI/litellm) 构建，提供一个完全 OpenAI 兼容的端点，解决 DeepSeek API 的兼容性问题，并集成 in-process 提示词优化技巧。

---

## 架构

```
客户端 (OpenAI SDK / Anthropic SDK) → DeepProxy (:8000 / :8001)
  ├─ [兼容层] 参数过滤 / 老模型别名 / reasoning / 错误映射 / Anthropic↔OpenAI 翻译
  ├─ [模型层] OpenRouter 风格 /v1/models（真实定价 / 上下文长度 / 仿冒别名）
  ├─ [优化层] 内建 skills（A/B/C/D 四组，0 额外 LLM 调用）
  │            + LLM 压缩（首次调一次，结果磁盘缓存复用）
  │            + 动态短段注入（场景化 PUA-substance 提示词）
  └─ [路由层] LiteLLM → DeepSeek API (api.deepseek.com)
```

## 解决的问题

| 问题 | 说明 |
|------|------|
| **参数兼容** | 自动过滤 DeepSeek 不支持的旧 OpenAI 参数（`functions` / `user`），避免 400 |
| **V4 别名层** | `deepseek-chat` / `deepseek-reasoner` 自动映射到 `deepseek-v4-flash` 并隐含正确的 `thinking.type` |
| **Reasoning 处理** | 保留 `reasoning_content`，多轮缓存自愈；模型剥离时从原始对象兜底恢复 |
| **错误映射** | 将 DeepSeek/LiteLLM 错误转换为标准 OpenAI 格式；429/5xx 指数退避重试 |
| **提示词优化** | 内建 15+ 廉价 skills（通用风格 / 反幻觉 / 上下文 / 消息转换），全 in-process，0 额外 LLM 调用 |
| **Anthropic 兼容** | 将 Anthropic Messages API 请求转换为 OpenAI 格式路由到 DeepSeek，支持流式和非流式 |
| **模型列表** | OpenRouter 风格 `/v1/models`，含真实 USD 定价、上下文长度、仿冒别名映射 |
| **克隆模型** | 将 pro/opus/codex 等仿冒模型别名映射到对应的 DeepSeek 实际模型 |

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

复制配置模板并编辑：

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，设置 deepseek.api_key
```

完整配置项见 [`config.example.yaml`](config.example.yaml)。关键结构：

```yaml
# 双端口绑定不同采样 profile
host: "0.0.0.0"
coding_port: 8000          # → precise_sampling
writing_port: 8001         # → creative_sampling

deepseek:
  api_key: ""                          # 填入你的 DeepSeek API 密钥
  api_base: "https://api.deepseek.com"

optimization:
  enabled: true
  compress_skills: true                # LLM 压缩 + 磁盘缓存
  dynamic_baskets: true                # 场景化中文短段注入
  # ... 完整 skills 开关见 config.example.yaml

creative_sampling:
  temperature_min: 0.90
  temperature_max: 1.20
  top_p_min: 0.90
  top_p_max: 0.97
  # presence_penalty / frequency_penalty 略

precise_sampling:
  temperature_min: 0.25
  temperature_max: 0.45
  top_p_min: 0.95
  top_p_max: 0.95
  presence_penalty_min: 0.0
  presence_penalty_max: 0.0
  frequency_penalty_min: 0.0
  frequency_penalty_max: 0.0
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
| `DEEPPROXY_RELOAD` | 热重载模式（仅 coding_port 生效） | `false` |

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全 (OpenAI 完全兼容) |
| `/v1/messages` | POST | Anthropic Messages API 兼容（请求被转换为 OpenAI 格式后路由到 DeepSeek） |
| `/v1/models` | GET | 列出可用模型（OpenRouter 风格，含定价/上下文长度/仿冒别名） |
| `/health` | GET | 健康检查 |

> 注：FIM (`/v1/completions`) 已下线——DeepSeek 官方 FIM 端点不支持 reasoning，需 FIM 的客户端请直连 DeepSeek `/beta/completions`。

## 提示词优化（Skills Pipeline）

所有优化在请求管道内顺次执行，全 in-process，无额外 LLM 调用（压缩器除外）。

**按通用程度分四组：**

### A. 通用风格 skills（每请求激活）
- `avoid_negative_style` — 禁说教套话
- `assume_good_intent` — 合理意图假设
- `instruction_priority` — system 最高权威
- `independent_analysis` — 自主推理（反谄媚）
- `reason_genuinely` — 真实推理，禁进度/时间幻觉
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
首次请求时，将所有 skills + 用户 system prompt 合并，调一次 LLM 压缩到最短同义版，按 `sha256(version + model + text)` 持久化到磁盘缓存文件。后续相同配置的请求直接命中缓存，0 上游调用。`inject_date` 使日期混入缓存键，每日自动刷新。

## 项目结构

```
deep_proxy/
├── deep_proxy/
│   ├── __init__.py              # 包标识（__version__ = "0.1.0"）
│   ├── main.py                  # FastAPI 应用与端点
│   ├── server.py                # 启动入口（双端口绑定）
│   ├── router.py                # 核心路由器（请求/响应生命周期）
│   ├── config.py                # Pydantic 配置模型
│   ├── litellm_client.py        # LiteLLM 调用封装（流式/非流式）
│   ├── models_list.py           # OpenRouter 风格 /v1/models 构建器
│   ├── deepseek_models.py       # 真实模型列表 + 仿冒别名映射
│   ├── deepseek_pricing.py      # USD / CNY 定价数据
│   ├── clone_models.py          # 仿冒模型条目生成
│   ├── utils.py                 # 共享工具函数（8 个）
│   ├── compatibility/
│   │   ├── __init__.py
│   │   ├── deepseek_fixes.py    # 模型名规范化 / 别名映射 / stream_options 清理
│   │   ├── reasoning_handler.py # reasoning_content 处理 + 服务端缓存
│   │   ├── error_mapper.py      # 参数过滤 + 错误码映射
│   │   └── anthropic_translator.py # Anthropic Messages API ↔ OpenAI 翻译层
│   └── optimization/
│       ├── __init__.py          # 编排入口（apply_cheap_optimizations）
│       ├── compressor.py        # LLM-based system prompt 压缩器
│       ├── skills_general.py    # 文本常量 + 辅助函数（A/B/C 组 skills）
│       ├── skills_transform.py  # 消息转换 skills（D 组：RE2/CoT/readurls）
│       ├── dynamic_baskets.py   # 场景化中文短段注入
│       └── silly_priming.py     # 无厘头 expert priming
├── tests/                       # pytest 套件（197 项测试）
│   ├── conftest.py              # 共享 fixtures
│   ├── test_router_pipeline.py  # 核心管道测试
│   ├── test_reasoning_handler.py
│   ├── test_cheap_optimizations.py
│   ├── test_compressor.py
│   ├── test_creative_sampling.py
│   ├── test_precise_sampling.py
│   ├── test_dual_port_profiles.py
│   ├── test_dynamic_baskets.py
│   ├── test_silly_priming.py
│   ├── test_list_models.py
│   ├── test_param_filtering.py
│   ├── test_deepseek_fixes.py
│   ├── test_retry.py
│   ├── test_anthropic_endpoint.py
│   ├── test_streaming_done.py
│   └── integration/             # 集成测试（需真实 API key）
├── config.yaml                  # 默认配置文件
├── config.example.yaml          # 配置模板
├── prompt_cache.json            # 提示词压缩磁盘缓存
├── requirements.txt             # Python 依赖
├── pytest.ini                   # pytest 配置
├── start.bat                    # Windows 启动脚本
├── QWEN.md                      # 开发上下文指南
└── CLAUDE.md                    # QWEN.md 的硬链接
```

## 开发

```bash
# 热重载模式（代码修改后自动重启）
set DEEPPROXY_RELOAD=true
python -m deep_proxy.server
```

## 许可

MIT
