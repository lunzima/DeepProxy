# Claude Code 一键接入 + 代理层 telemetry 过滤

**日期**：2026-05-19
**状态**：Design approved，等待 writing-plans

## 1. 背景与目标

### 1.1 用户问题

- Claude Code 接入 DeepProxy 需要手动配置 4–6 个环境变量（`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` / `CLAUDE_CODE_ATTRIBUTION_HEADER` 等），文档分散、容易遗漏、`AUTH_TOKEN` 与 `API_KEY` 混淆。
- Claude Code 2.1.42 在 system prompt 头部注入含 session hash 的 `x-anthropic-billing-header`，每次新会话破坏前缀缓存（详见 `blog.deepai.wiki/posts/claude-code-cache-pitfall/`）。客户端有 `CLAUDE_CODE_ATTRIBUTION_HEADER=false` 可关，但需依赖用户正确配置；若忘配 / 跨机器 / 客户端版本变化导致开关失效，代理层无兜底。

### 1.2 本次目标

- **目标 A**：提供 Windows 一键脚本，永久写入用户级环境变量（无需管理员），覆盖 Claude Code 接入 DeepProxy 所需的全部变量。
- **目标 B**：代理层独立剥离 `x-anthropic-*` telemetry header 行（不依赖客户端配置），让缓存稳定性与压缩开关、客户端版本解耦。

### 1.3 非目标

- 不做 macOS / Linux 脚本（用户当前需求仅 Windows；后续可加）。
- 不动 HTTP 请求的真实 header，只处理消息体里的伪 header 文本。
- 不引入新的 LLM 调用或路由器。
- 不改 `CLAUDE_CODE_EFFORT_LEVEL` 等高级运行时变量（不在本次范围）。

## 2. 架构概览

```
[用户]
  │
  ├─ 双击 tools\setup_claude_code_env.bat ─→ 调 pwsh ─→ setup_claude_code_env.ps1
  │     ├─ 读 config.yaml（host / coding_port / writing_port / api_key）
  │     ├─ setx <USER 级> ANTHROPIC_BASE_URL / AUTH_TOKEN / MODEL / SMALL_FAST_MODEL / CLAUDE_CODE_ATTRIBUTION_HEADER
  │     ├─ 当前 session $env:* 同步写入
  │     └─ 打印总结 + 提示"打开新终端"
  │
  └─ claude code 运行 ─→ 发请求到 DeepProxy
        ↓
   DeepProxy router.prepare_request
        ↓
    [新] step 3.5: strip_telemetry_from_messages
        ├─ 扫描 system 消息 + 首条 user 消息
        └─ 删除 ^x-anthropic-[a-z-]+:.*$ 行（multiline + 大小写不敏感）
        ↓
    skills 优化 / 压缩 / 升格 / LiteLLM → DeepSeek
```

两个组件互相**独立**：脚本只动 OS 环境变量；过滤层只动消息体。任一可单独 ship、单独回滚。

## 3. 组件设计

### 3.1 Windows 一键脚本

#### 3.1.1 文件结构

```
tools/
  setup_claude_code_env.ps1     # 主脚本（PowerShell 5+）
  setup_claude_code_env.bat     # 双击启动器（调 pwsh / powershell）
```

`.bat` 启动器仅做两件事：
1. 切到脚本目录
2. `powershell -ExecutionPolicy Bypass -File "%~dp0setup_claude_code_env.ps1" %*`

参数透传（`-DryRun` / `-Uninstall` / `-Writing` / `-Force`）。

#### 3.1.2 脚本流程（PowerShell）

```
1. Parse args:  [switch]$DryRun, [switch]$Uninstall, [switch]$Writing, [switch]$Force
2. Resolve config.yaml 路径（脚本所在目录的上一级；缺失走默认值）
3. 解析 config.yaml：
     - host, coding_port, writing_port, api_key
     - host == "0.0.0.0" → 改写为 "127.0.0.1"（OS 客户端不能向 0.0.0.0 发请求）
     - 端口：默认 coding_port；若 -Writing 则 writing_port
4. 计算环境变量值：
     ANTHROPIC_BASE_URL  = "http://<host>:<port>"
     ANTHROPIC_AUTH_TOKEN = config.api_key (非空) | "dummy"
     ANTHROPIC_MODEL     = "deepseek-v4-pro[1m]"
     ANTHROPIC_SMALL_FAST_MODEL = "deepseek-v4-flash"
     CLAUDE_CODE_ATTRIBUTION_HEADER = "false"
     ANTHROPIC_API_KEY   → DELETE（避免与 AUTH_TOKEN 冲突）
5. 若 -DryRun：打印表格，不写。退出 0。
6. 若 -Uninstall：删除本脚本**设置过**的 5 个变量（`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` / `CLAUDE_CODE_ATTRIBUTION_HEADER`）；**不**触碰 `ANTHROPIC_API_KEY`（无法恢复原值，且用户可能在其它工具上手动重设）。退出 0。
7. 冲突检测：
     - 现有 ANTHROPIC_API_KEY 已设且非空 → 警告并提示移除
       · 有 -Force → 直接移除
       · 无 -Force → 终端交互式 [Y/n] 确认；非 TTY 默认 N + 报错退出
8. 写入：
     - 用户级永久：[Environment]::SetEnvironmentVariable($name, $value, "User")
       （等价 setx，但能写 $null 删除变量，且不触发 1024 字符截断警告）
     - 当前 session：$env:NAME = $value
9. 打印总结表（变量名 + 掩码后的值），末尾醒目提示：
     "已写入用户级环境变量。当前终端可立即使用；其它已开终端需重启后才会读到新值。"
10. 退出码：0=成功，1=参数错误，2=配置/IO 错误，3=用户拒绝覆盖
```

#### 3.1.3 关键决策

- **用 `[Environment]::SetEnvironmentVariable` 而非 `setx`**：行为等价（写 HKCU\Environment 后广播 `WM_SETTINGCHANGE`），但能写 `$null` 删除变量，且无 1024 字符截断。已被微软官方推荐替代 `setx`。
- **不引入 admin / Machine 级**：单用户项目；`User` 级 scope 对单一 Windows 用户已足够，且无需 UAC。
- **`$env:` 与持久化同步**：让用户**当前**已开 shell 也立即可用，避免"明明设了为什么不生效"的混淆。
- **掩码 token 输出**：参考 `main.py:_mask`，保留前 6 + 后 2 字符。
- **`api_key` 留空时回退 `"dummy"`**：DeepProxy 的 `api_key` 是入口鉴权 key，留空意味不校验；Claude Code 客户端必须发非空 `ANTHROPIC_AUTH_TOKEN`，否则报错。

#### 3.1.4 YAML 解析

不引入 `powershell-yaml` 模块依赖（用户可能没装）。采用**正则提取**有限子集：

- 仅解析顶层标量：`host`、`coding_port`、`writing_port` 和嵌套一层 `api_key`（顶层 `api_key`，不动 `deepseek.api_key`）
- 解析失败 → 警告 + 用默认值（`host=127.0.0.1`，`coding_port=8000`，`writing_port=8001`，`api_key=null`）

正则示例：`^api_key:\s*["']?([^"'\s#]+)["']?\s*(#.*)?$`，对每个键独立扫描。

### 3.2 代理层 telemetry 过滤

#### 3.2.1 新文件 `deep_proxy/optimization/strip_telemetry.py`

```python
import re

# 与 compressor 共用的动态 header 剥离正则
_TELEMETRY_HEADER_RE = re.compile(
    r'^x-anthropic-[a-z-]+:.*$',
    re.MULTILINE | re.IGNORECASE,
)

def strip_telemetry_from_text(text: str) -> str:
    """删除文本中匹配 x-anthropic-* header 模式的整行。"""
    if not text:
        return text
    return _TELEMETRY_HEADER_RE.sub("", text)

def strip_telemetry_from_messages(messages: list[dict]) -> None:
    """就地修改 messages：剥离 system 消息 + 首条 user 消息中的 telemetry header 行。

    - system 消息：content 可能为 str / list[dict]；list 时遍历 text 块
    - 首条 user 消息：Claude Code 也可能把 header 注入这里（防御性覆盖）
    - 容错：非预期类型静默跳过，不抛异常
    """
```

#### 3.2.2 compressor 复用

```python
# deep_proxy/optimization/compressor.py
from .strip_telemetry import _TELEMETRY_HEADER_RE as _DYNAMIC_HEADERS_RE
```

去掉本地 `_DYNAMIC_HEADERS_RE` 定义，单一来源避免漂移。

#### 3.2.3 接线 `router.prepare_request`

在现有 7 步管道中**新增 step 3.5**：

```
1. Legacy alias → V4 + 隐式 thinking
2. thinking.reasoning_effort=max 注入
3. 采样默认值
3.5 [新] strip_telemetry_from_messages（受 cfg.optimization.strip_client_telemetry 控制）
4. strip_unsupported_params
5. ensure_reasoning_content_persistence
6. sanitize_stream_options
7. apply_cheap_optimizations
8. 动态短段注入
9. 无厘头 expert priming
```

**位置选择理由**：
- 早于 skills/压缩：让所有下游看到干净文本，包括压缩缓存 key 计算（与 compressor.py 当前的 `_normalize` 形成纵深防御）
- 晚于采样 / alias：那两步不读消息体内容，过滤顺序无关
- 与 `strip_unsupported_params` 同属"清理"语义但作用对象不同（前者 params，后者 messages），分开独立步骤更清晰

#### 3.2.4 配置开关

```yaml
optimization:
  strip_client_telemetry: true   # 默认 true：剥离客户端 telemetry header 行
                                 # （x-anthropic-billing-header 等；稳定缓存前缀）
```

加到 `ProxyConfig.optimization` 的 Pydantic 模型，带中文 `Field(description=...)`。

`config.example.yaml` 在 "A. 通用风格 skills" 之前的"清理"段加注释。

### 3.3 数据流：以 Claude Code 2.1.42 请求为例

**入站 system**：

```
x-anthropic-billing-header: cc_version=2.1.42.abc; cc_entrypoint=claude-code; cch=00000;
You are Claude Code, ...
```

**step 3.5 后**：

```

You are Claude Code, ...
```

（首行变空行；空行对下游 LLM 语义无影响，无需额外 `\n` 整理）

**进入 step 7 (compressor)**：compressor 仍跑 `_normalize`，但此时已 no-op（防御性双层兜底，零额外成本）。压缩缓存 key 稳定，跨会话命中。

**下游 DeepSeek 看到**：稳定前缀，激活其 context cache。

## 4. 错误处理

| 场景 | 行为 |
|------|------|
| 脚本：config.yaml 不存在 | 警告 + 用默认值继续 |
| 脚本：config.yaml 解析失败 | 警告 + 用默认值继续 |
| 脚本：现有 `ANTHROPIC_API_KEY` 非空 | 交互确认；`-Force` 跳过；非 TTY 默认拒绝 |
| 脚本：`SetEnvironmentVariable` 抛 IO 异常 | 打印异常 + 退出码 2 |
| 过滤：messages 非 list | 静默 no-op |
| 过滤：content 类型非 str / list[dict] | 该消息跳过，不抛 |
| 过滤：list[dict] 中某块缺 `text` 字段 | 该块跳过，不抛 |
| 过滤：整体异常 | try/except 包住，warning 日志，原文返回（不阻塞主请求） |

## 5. 测试计划

| 测试 | 文件 | 覆盖 |
|------|------|------|
| 单元 | `tests/test_strip_telemetry.py`（新建） | (1) 单行剥离 (2) 多行剥离 (3) 大小写不敏感 (4) 未知 `x-anthropic-` 子名也匹配 (5) 不误删 `x-anthropic-foo` 作为正文出现的情况（仅整行匹配） (6) system 为 str / list[dict] 两种形态 (7) user 首条剥离 (8) messages=None / [] no-op |
| 集成 | `tests/test_router_pipeline.py` 增 case | `prepare_request` 启用过滤后 system 不含 `x-anthropic-*`；关闭开关后保留 |
| 回归 | `tests/test_compressor.py` 调整 | compressor 引用共享正则后，原有归一化测试仍通过 |
| 脚本验证 | 不写 Python 测试；spec 附手动验证清单 | (1) `setup_claude_code_env.bat -DryRun` 打印预期表格 (2) 实跑后 `[Environment]::GetEnvironmentVariable("ANTHROPIC_BASE_URL","User")` 返回正确值 (3) `-Uninstall` 后变量被清空 (4) 新开终端能读到值 |

## 6. 文档更新

- `README.md`：新增 "Windows 一键接入 Claude Code" 段落，给出 3 步：(1) 启动 DeepProxy (2) 双击 `tools\setup_claude_code_env.bat` (3) 重开终端跑 `claude`
- `CLAUDE.md`：
  - Request Pipeline 列表加入 step 3.5
  - Skills Pipeline 简单提及 telemetry 过滤作为"清理 skill"
- `config.example.yaml`：加 `strip_client_telemetry: true` 项 + 中文注释
- 不写独立的 `docs/` 用户指南（README 段足以）

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Claude Code 未来改 telemetry header 命名（如 `x-claude-*`） | 正则按前缀匹配 `x-anthropic-*`；新前缀需手动加，未来若发生在新增模式时一并扩展 |
| 用户系统 PowerShell 版本过老 | 项目仅支持 Windows 10/11（与 DeepProxy 其它部分一致）；这些系统内置 PowerShell 5.1+，无需探测降级 |
| 过滤正则误删用户消息正文中合法的 `x-anthropic-foo:` 行 | 仅匹配**行首**且为标准 header 形态；用户在代码块/文档中引用此类字符串需用缩进或非行首位置，与正常使用习惯冲突极低；接受此风险 |
| 删除 `ANTHROPIC_API_KEY` 触发用户其它工具（非 Claude Code）异常 | 默认交互式确认；提供 `-Force` 用于 CI；`-Uninstall` 不强制删除 `ANTHROPIC_API_KEY`（仅删本脚本设置的变量） |

## 8. 不实现 / 留作后续

- macOS / Linux 脚本
- 全局机器级（`Machine` scope）变量
- 自动启动/检测 DeepProxy 服务（脚本仅配置环境变量，不动 service）
- 多 profile 切换（如 `coding` / `writing` 快速切换的独立子命令）
- `CLAUDE_CODE_EFFORT_LEVEL=max` 等运行时高级变量（用户可手动加）

## 9. 验收标准

1. 新机器（已装 DeepProxy + Python）双击 `setup_claude_code_env.bat`，无报错，输出包含所有目标变量。
2. 新开 PowerShell，`echo $env:ANTHROPIC_BASE_URL` 返回 `http://127.0.0.1:8000`。
3. `claude` 直接运行可连通 DeepProxy（前提：DeepProxy 已启动）。
4. 发起包含 `x-anthropic-billing-header: ...` 的请求，DeepProxy 日志 / 下游 DeepSeek 调用不再含该行。
5. `tests/test_strip_telemetry.py` 全绿；`tests/test_compressor.py` 不回归。
6. `-DryRun` 不写变量；`-Uninstall` 后变量为空。
