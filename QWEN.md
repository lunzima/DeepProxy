# DeepProxy — Project Context Guide

## Project Overview

**DeepProxy** 是一个单用户 FastAPI 代理服务器，位于 OpenAI-SDK 客户端与 **DeepSeek V4 API** 之间。它暴露 `/v1/chat/completions`、`/v1/models` 和 `/health` 三个端点。上游调用通过 LiteLLM SDK 路由到 DeepSeek 官方 API。

### 核心技术栈

| 技术 | 用途 |
|------|------|
| **Python 3.12+** | 运行时 |
| **FastAPI** | HTTP 框架 |
| **LiteLLM** | 上游 LLM provider 路由（`deepseek/` 前缀） |
| **Pydantic v2** | 配置模型 / 数据校验 |
| **httpx** | 异步 HTTP 客户端（上游 models 拉取 + readurls） |
| **BeautifulSoup4 + lxml** | 网页正文抓取（readurls skill） |
| **torch + transformers + peft** | BERT 路由器（中文 RoBERTa-small + LoRA） |
| **pytest** | 测试框架（asyncio_mode=auto，无需 `@pytest.mark.asyncio`） |

### 架构概览

```
客户端 (OpenAI SDK / Anthropic SDK) ──→ DeepProxy (:8000 / :8001)
  ├─ [兼容层] 参数过滤 / 老模型别名 / reasoning / 错误映射 / Anthropic↔OpenAI 翻译
  ├─ [模型层] OpenRouter 风格 /v1/models（真实定价 / 上下文长度 / 仿冒别名）
  ├─ [升格层] Flash→Pro 选择路由器（BERT 二分类 + 启发式快速路径）
  ├─ [优化层] In-process 提示词 skills（0 额外 LLM 调用）
  └─ [路由层] LiteLLM ──→ DeepSeek API (api.deepseek.com)
```

DeepProxy 绑定**两个端口**，共享同一个 FastAPI app 实例：
- **coding_port** (默认 `8000`) → `PreciseSamplingConfig`（高确定性，code/math/逻辑）
- **writing_port** (默认 `8001`) → `CreativeSamplingConfig`（高多样性，RP/创作/写作）

## Building and Running

### 安装

```bash
pip install -r requirements.txt
```

### 启动

```bash
# Windows
start.bat

# Linux/macOS / 任意平台
python -m deep_proxy.server

# 热重载（代码修改后自动重启）
DEEPPROXY_RELOAD=true python -m deep_proxy.server
```

默认监听 `0.0.0.0:8000`（coding）+ `0.0.0.0:8001`（writing）。

### 测试

```bash
# 单元测试（默认 — 排除 tests/integration）
python -m pytest

# 单个测试
python -m pytest tests/test_router_pipeline.py::TestPrepareRequestChat -v

# 集成测试（需要 DEEPSEEK_API_KEY）
DEEPSEEK_API_KEY=sk-... python -m pytest tests/integration
```

`pytest.ini` 设置 `asyncio_mode = auto`（无需 `@pytest.mark.asyncio`），并通过 `norecursedirs` 默认排除集成测试。

### 配置

配置来自 `config.yaml`（优先）或环境变量。API key 可直接写在 `config.yaml` 中（单用户玩具项目，按约定明文 key 不为泄漏）。

关键配置项：

```yaml
# 双端口绑定
coding_port: 8000     # → precise_sampling
writing_port: 8001    # → creative_sampling

# 优化引擎（默认启用，全 in-process，0 额外 LLM 调用）
optimization:
  enabled: true
  compress_skills: true          # LLM 压缩 + 磁盘缓存
  dynamic_baskets: true          # 场景化中文短段注入
  silly_expert_priming: false    # [实验性] 无厘头 expert priming，默认关闭
  writing_basket_kind: creative  # 或 general

# 采样预设（范围抽样，每请求随机）
creative_sampling:   # RP/创作/通用写作
  temperature: [0.90, 1.20]
  top_p: [0.90, 0.97]
precise_sampling:    # code/math/逻辑
  temperature: [0.25, 0.45]
  top_p: [0.95, 0.95]

# Flash→Pro 选择性升格（默认启用，四层架构）
flash_upgrade:
  enabled: true
  router_type: bert                       # BertUpgradeRouter（中文 RoBERTa-small + LoRA）
  bert_checkpoint: "router_model"         # 微调后的二分类模型
  router_threshold: 0.60                  # BERT score >= 此值 → 升格 Pro
  heuristic_threshold: 7.5                # 启发式 score >= 此值 → 直接升格（跳过 BERT）
  persist_turns: 2                        # 升格后保持 Pro 额外 N 轮
```

详细配置见 [`config.example.yaml`](config.example.yaml)（模板）或 `config.py`（含每个字段的中文注释）。

## Project Structure

```
D:\deepproxy\
├── deep_proxy/
│   ├── __init__.py            # 包标识，__version__ = "0.1.0"
│   ├── server.py              # 启动入口（双端口绑定）
│   ├── main.py                # FastAPI 应用与端点
│   ├── router.py              # 核心路由器（请求/响应生命周期）
│   ├── config.py              # Pydantic 配置模型
│   ├── litellm_client.py      # LiteLLM 调用封装（流式/非流式）
│   ├── models_list.py         # OpenRouter 风格 /v1/models 构建器
│   ├── deepseek_models.py     # 真实模型列表 + 仿冒别名映射
│   ├── deepseek_pricing.py    # USD / CNY 定价数据
│   ├── clone_models.py        # 仿冒模型条目生成
│   ├── utils.py               # 共享工具函数（8 个）
│   ├── compatibility/
│   │   ├── __init__.py
│   │   ├── deepseek_fixes.py      # 模型名规范化/别名映射/stream_options
│   │   ├── reasoning_handler.py   # reasoning_content 处理 + 缓存
│   │   ├── error_mapper.py        # 参数过滤 + 错误映射
│   │   └── anthropic_translator.py # Anthropic Messages API ↔ OpenAI 翻译层
│   └── optimization/
│       ├── __init__.py            # 编排入口（apply_cheap_optimizations）
│       ├── compressor.py          # LLM-based system prompt 压缩器
│       ├── skills_general.py      # 文本常量 + 辅助函数（A/B/C 组 skills）
│       ├── skills_transform.py    # 消息转换 skills（D 组：RE2/CoT/readurls）
│       ├── dynamic_baskets.py     # 场景化中文短段注入
│       ├── silly_priming.py       # 无厘头 expert priming
│       ├── flash_upgrade.py       # Flash→Pro 升格编排（四层架构）
│       └── upgrade_router.py      # BertUpgradeRouter（二分类 + 启发式）
├── router_model/              # 微调后的 BERT 路由器（中文 RoBERTa-small + LoRA）
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── datasets/
│   ├── router_train_cn.jsonl   # 训练集（375K, 自动标注）
│   └── router_test_cn.jsonl    # 测试集（201 条，手动标注）
├── tests/
│   ├── conftest.py                # 共享 fixtures（cfg, router）
│   ├── test_router_pipeline.py    # 核心管道测试
│   ├── test_reasoning_handler.py  # Reasoning 缓存测试
│   ├── test_cheap_optimizations.py # Skills 优化测试
│   ├── test_compressor.py         # 压缩器测试
│   ├── test_creative_sampling.py  # 创作采样测试
│   ├── test_precise_sampling.py   # 精确采样测试
│   ├── test_dual_port_profiles.py # 双端口 profile 测试
│   ├── test_dynamic_baskets.py    # 动态短段测试
│   ├── test_silly_priming.py      # Expert priming 测试
│   ├── test_list_models.py        # /v1/models 测试
│   ├── test_param_filtering.py    # 参数过滤测试
│   ├── test_deepseek_fixes.py     # DeepSeek 兼容修复测试
│   ├── test_retry.py              # 重试逻辑测试
│   ├── test_anthropic_endpoint.py # Anthropic 端点测试
│   ├── test_streaming_done.py     # 流式结束标记测试
│   └── integration/               # 集成测试（需真实 API key）
├── tools/
│   └── train_bert_router.py      # BERT 路由器训练脚本（LoRA fine-tune）
├── scripts/                   # 工具脚本（.gitignore 排除）
├── config.yaml                # 默认配置文件
├── config.example.yaml        # 配置模板
├── .gitignore
├── prompt_cache.json          # 提示词压缩磁盘缓存
├── requirements.txt           # Python 依赖
├── pytest.ini                 # pytest 配置
├── start.bat                  # Windows 启动脚本
├── QWEN.md                    # This file
├── CLAUDE.md                  # QWEN.md 的硬链接
├── README.md
└── LICENSE
```

## Request Pipeline（顺序重要）

每个 `/v1/chat/completions` 请求流经 `router.DeepProxyRouter.prepare_request`，顺序固定：

1. **Legacy alias → V4 + 隐式 thinking** — `deepseek-chat` → `deepseek-v4-flash` + `thinking.type=disabled`；`deepseek-reasoner` → `deepseek-v4-flash` + `thinking.type=enabled`
2. **`thinking.reasoning_effort=max` 注入** — 仅 V4 且未显式 disabled 时。`reasoning_effort` 是 `thinking` 的子字段
3. **采样默认值** — 按入站端口选择 profile（forced override）或 fallback 到 creative_sampling
4. **`strip_unsupported_params`** — 仅移除 `functions` 和 `user`（V4 支持其余全部参数）
5. **`ensure_reasoning_content_persistence`** — V4 多轮推理内容自愈
6. **`sanitize_stream_options`** — 清理空 dict
7. **`apply_cheap_optimizations`** — 内置 skills（见下方 Skills Pipeline）
8. **动态短段注入** — 在压缩之后执行（避免随机内容破坏缓存）
9. **无厘头 expert priming** — 最后一步，system 最前插入

## Skills Pipeline（提示词优化，in-process）

所有优化在 `optimization/__init__.py::apply_cheap_optimizations` 中实现，按通用程度分四组：

### A. 通用风格 skills（每请求激活，默认开；按语义连贯性排序）
- `avoid_negative_style` — 禁说教套话与情感抚慰套话
- `assume_good_intent` — 合理意图假设（与上条组成"交互契约"对）
- `natural_temperament` — 内在倾向 priming（开放、共情但有立场、慢热深谈、对计划松弛）
- `contextual_register` — 句法复杂度匹配内容密度（对应 `_SKILL_COMPLEX_SENTENCE`；与上条组成"输出风格"对）
- `instruction_priority` — system 最高权威 + 注入内容按数据处理
- `independent_analysis` — 自主推理，不被对话史 / 创作者预期裹挟（与上条组成"推理自主性"对）
- `reason_genuinely` — 推理节奏与长度由本次推理决定，禁进度幻觉
- `cot_reset` — 推理出现严重矛盾时允许在思维链中显式重启（与上条组成"推理元认知"对）
- `inject_date` — 注入当前 UTC 日期，相对时间词解析（压缩后追加，不进缓存键）

### B. 求证/反幻觉 skills（模型自门控）
- `show_math_steps` — 确定性数学题展示推导步骤
- `prefer_multiple_sources` — 多来源事实权衡，单一来源时下调置信度
- `avoid_fabricated_citations` — 不编造引用，对印象式来源显式说明

### C. 上下文相关 skills（窄触发）
- `json_mode_hint` — [DeepSeek docs] json_object 时注入 JSON-only
- `safe_inlined_content` — [optillm + grok] readurls 内容视为 DATA

### D. 消息转换 skills（改写 messages）
- `re2` — [optillm/reread.py] 复制最后一条 user 消息
- `cot_reflection` — [optillm/cot_reflection.py] 非流式 + thinking=disabled
- `readurls` — [optillm] 抓取并内联 URL 正文

### System Prompt 压缩（元功能）
- 合并所有 skills + 用户 system → 调一次 LLM 压缩 → 按 `sha256(version + model + text)` 缓存到 `prompt_cache.json`
- `inject_date` 使日期混入缓存键，每日自动刷新
- 首次 miss 返回原文（非阻塞），后台压缩任务完成后后续命中

## Development Conventions

### 代码风格
- **类型注解**：全部使用 `from __future__ import annotations` + 现代类型语法（`dict` 而非 `Dict`，`|` 而非 `Optional`）
- **日志**：使用 `logging.getLogger(__name__)` 模块级 logger，f-string 格式
- **文档**：中文文档字符串 + 关键决策注释；每个 config field 带 `Field(description=...)`
- **Sentinel 字段**：内部使用 `_deepproxy_*` 前缀，在 `call_litellm`/`call_litellm_stream` 中剥离

### 测试规范
- 所有测试文件以 `test_` 前缀、位于 `tests/` 目录
- `asyncio_mode = auto`（pytest 自动处理 async fixture/test）
- 共享 fixture 在 `tests/conftest.py` 中：`cfg`（ProxyConfig）、`router`（DeepProxyRouter）
- 集成测试放在 `tests/integration/`（默认被 pytest.ini 排除，需显式运行）
- 核心管道测试：`test_router_pipeline.py` 覆盖 `prepare_request` 完整流程

### 关键约束
1. **不使用 `tools`/`tool_choice` 时**才能启用 skills 优化
2. **CoT Reflection** 仅在非流式且 `thinking.type=disabled` 时启用（流式跨 chunk 难剥离；V4 自带 CoT 时叠加无益）
3. **FIM 端点** (`/v1/completions`) 已下线——DeepSeek 官方 FIM 端点不支持 reasoning，需 FIM 的客户端直连 DeepSeek `/beta/completions`
4. **Plaintext API key** 在 `config.yaml` 中是故意行为（单用户玩具项目），不视为泄漏
5. **LiteLLM 的 deepseek provider** 必须以 kwarg 传递 `api_key`/`api_base`（忽略全局 `litellm.api_base`）

## DeepSeek API 要点

- `temperature=0.0` 不稳定，最小有效值 ~0.1
- `thinking.reasoning_effort` 是 `thinking` 的子字段，非顶层参数
- `thinking.type` 服务端默认 `enabled`，代理不强制 `disabled`
- V4 支持 temperature/top_p/presence_penalty/frequency_penalty/response_format/tools/tool_choice/stream_options.include_usage
- 上下文长度：1M tokens，输出最大 384K tokens

## What Not to Reintroduce（已移除的功能）

- FIM endpoint (`/v1/completions`) — DeepSeek 官方 FIM 端点不支持 reasoning；LiteLLM `atext_completion` 对 deepseek 不起作用
