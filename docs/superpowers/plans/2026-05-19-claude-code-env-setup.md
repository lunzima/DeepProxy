# Claude Code 一键接入 + 代理层 telemetry 过滤 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户一键完成 Windows 下 Claude Code 接入 DeepProxy 的所有环境变量配置（永久、无需管理员），并在代理层独立剥离 `x-anthropic-*` telemetry header 行，让缓存稳定性与客户端配置解耦。

**Architecture:**
1. 新增 `deep_proxy/optimization/strip_telemetry.py` 暴露共享正则与剥离函数；`compressor.py` 复用该正则避免漂移；`router.prepare_request` 在模型名归一化后、Flash→Pro 升格判定前调用剥离函数，让所有下游（升格哈希 / skills / 压缩缓存 key）都看到干净文本。
2. 新增 `tools/setup_claude_code_env.{ps1,bat}` 使用 `[Environment]::SetEnvironmentVariable($name, $value, "User")` 写入用户级永久环境变量并同步当前 session（`$env:VAR=value`），无需管理员。

**Tech Stack:** Python 3.12+ / pytest / pydantic v2 / re / PowerShell 5.1+ / Windows batch

**Spec:** `docs/superpowers/specs/2026-05-19-claude-code-env-setup-design.md`

---

## File Structure

**New:**
- `deep_proxy/optimization/strip_telemetry.py` — 共享正则 + `strip_telemetry_from_text` / `strip_telemetry_from_messages`
- `tests/test_strip_telemetry.py` — 单元测试
- `tools/setup_claude_code_env.ps1` — 主脚本
- `tools/setup_claude_code_env.bat` — 双击启动器
- `docs/superpowers/plans/2026-05-19-claude-code-env-setup.md` — 本计划

**Modified:**
- `deep_proxy/optimization/compressor.py` — 移除本地 `_DYNAMIC_HEADERS_RE`，从 `strip_telemetry` 导入
- `deep_proxy/config.py` — 在 `OptimizationConfig` 顶部加 `strip_client_telemetry` 字段
- `deep_proxy/router.py` — 在 `prepare_request` 中调用过滤函数（model normalize 后、Flash→Pro 升格判定前）
- `tests/test_router_pipeline.py` — 加 router 集成测试
- `tests/test_compressor.py`（可能）— 若现有正则引用方式变化则同步
- `config.example.yaml` — 添加 `strip_client_telemetry: true` 项
- `CLAUDE.md` — Request Pipeline 列表加入 step 0c'（telemetry strip）
- `README.md` — 加 "Windows 一键接入 Claude Code" 段

---

### Task 1: telemetry 过滤模块 + 单元测试（TDD）

**Files:**
- Create: `deep_proxy/optimization/strip_telemetry.py`
- Create: `tests/test_strip_telemetry.py`

- [ ] **Step 1: 写测试文件（先红，单元测试驱动设计）**

```python
# tests/test_strip_telemetry.py
"""测试 telemetry header 行剥离。

覆盖：
- 单行 / 多行剥离
- 大小写不敏感
- 任意 x-anthropic-* 子名匹配
- 不误删非行首位置的同名字符串
- str / list[dict] 两种 system content 形态
- 首条 user 消息也覆盖
- None / [] / 缺失字段的 no-op 容错
"""
from __future__ import annotations

from deep_proxy.optimization.strip_telemetry import (
    _TELEMETRY_HEADER_RE,
    strip_telemetry_from_text,
    strip_telemetry_from_messages,
)


class TestStripText:
    def test_single_header_line_removed(self):
        text = (
            "x-anthropic-billing-header: cc_version=2.1.42.abc; cc_entrypoint=claude-code; cch=0;\n"
            "You are Claude Code."
        )
        out = strip_telemetry_from_text(text)
        assert "x-anthropic-billing-header" not in out
        assert "You are Claude Code." in out

    def test_multiple_x_anthropic_lines_removed(self):
        text = (
            "x-anthropic-foo: a\n"
            "x-anthropic-bar: b\n"
            "real content"
        )
        out = strip_telemetry_from_text(text)
        assert "x-anthropic-foo" not in out
        assert "x-anthropic-bar" not in out
        assert "real content" in out

    def test_case_insensitive(self):
        text = "X-Anthropic-Billing-Header: foo\nreal"
        out = strip_telemetry_from_text(text)
        assert "Billing-Header" not in out
        assert "real" in out

    def test_inline_mention_not_removed(self):
        # 仅匹配行首；正文里引用 "x-anthropic-foo: ..." 不应被吃掉
        text = "Quote: 'x-anthropic-foo: bar' as example"
        out = strip_telemetry_from_text(text)
        assert "x-anthropic-foo: bar" in out

    def test_empty_input(self):
        assert strip_telemetry_from_text("") == ""
        assert strip_telemetry_from_text(None) is None  # type: ignore[arg-type]

    def test_no_match_returns_unchanged(self):
        text = "Plain system prompt with no telemetry."
        assert strip_telemetry_from_text(text) == text


class TestStripMessages:
    def test_system_str_content_stripped(self):
        messages = [
            {"role": "system", "content": "x-anthropic-billing-header: foo\nYou are helpful."},
            {"role": "user", "content": "hi"},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-billing-header" not in messages[0]["content"]
        assert "You are helpful." in messages[0]["content"]

    def test_system_list_content_stripped(self):
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "x-anthropic-foo: bar\nrules"},
                {"type": "text", "text": "more rules"},
            ]},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-foo" not in messages[0]["content"][0]["text"]
        assert messages[0]["content"][1]["text"] == "more rules"

    def test_first_user_message_also_stripped(self):
        messages = [
            {"role": "user", "content": "x-anthropic-billing-header: foo\nactual question"},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-billing-header" not in messages[0]["content"]
        assert "actual question" in messages[0]["content"]

    def test_later_user_message_untouched(self):
        # 只清理首条 user（CC 注入位置）；后续 user 消息可能合法引用该字符串
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "x-anthropic-foo: should NOT be touched here"},
        ]
        strip_telemetry_from_messages(messages)
        assert "x-anthropic-foo" in messages[3]["content"]

    def test_empty_messages_noop(self):
        strip_telemetry_from_messages([])  # 不抛异常
        strip_telemetry_from_messages(None)  # type: ignore[arg-type]

    def test_non_dict_message_skipped(self):
        # 容错：遇到非 dict 项不抛
        messages = ["bogus", {"role": "system", "content": "x-anthropic-foo: bar\nok"}]
        strip_telemetry_from_messages(messages)  # type: ignore[arg-type]
        assert "x-anthropic-foo" not in messages[1]["content"]

    def test_non_str_content_left_alone(self):
        # content 既非 str 也非 list[dict]：静默跳过，不抛
        messages = [{"role": "system", "content": 12345}]
        strip_telemetry_from_messages(messages)  # type: ignore[list-item]
        assert messages[0]["content"] == 12345


class TestRegex:
    def test_regex_pattern_anchored_to_line_start(self):
        # 文档化共享正则的行为
        import re
        assert _TELEMETRY_HEADER_RE.flags & re.MULTILINE
        assert _TELEMETRY_HEADER_RE.flags & re.IGNORECASE
```

- [ ] **Step 2: 运行测试确认全部失败**

Run: `python -m pytest tests/test_strip_telemetry.py -v`
Expected: 全部 fail，错误 `ModuleNotFoundError: No module named 'deep_proxy.optimization.strip_telemetry'`

- [ ] **Step 3: 实现 `strip_telemetry.py`**

```python
# deep_proxy/optimization/strip_telemetry.py
"""客户端 telemetry header 行剥离。

针对 Claude Code 2.1.42+ 在 system prompt 头部注入 `x-anthropic-billing-header:`
这类含 session hash 的伪 header 行：每次新会话哈希变化破坏 prefix cache。
本模块只动消息体里的伪 header 文本，不动 HTTP 请求的真实 header。

设计：
- 单一来源的共享正则 `_TELEMETRY_HEADER_RE`（compressor.py 通过 import 复用）。
- `strip_telemetry_from_text`：纯函数，行首匹配 `x-anthropic-<name>:` 整行删除。
- `strip_telemetry_from_messages`：就地处理 system 消息 + 首条 user 消息。
- 容错优先于诊断：异常情况静默跳过，不阻塞主请求。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 匹配 Claude Code 等客户端在 system prompt 中嵌入的伪 header 行。
# - ^ + re.MULTILINE：仅行首匹配，避免误删正文中作为引用出现的字符串。
# - re.IGNORECASE：CC 历史版本曾用 X-Anthropic-* / x-anthropic-* 混合大小写。
_TELEMETRY_HEADER_RE = re.compile(
    r"^x-anthropic-[a-z-]+:.*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_telemetry_from_text(text: Any) -> Any:
    """剥离文本中匹配 telemetry header 模式的整行。

    - 非 str 输入原样返回（容错；调用方可能传 None / list 等）
    - 不合并连续空行：删除后留下的空行对下游 LLM 语义无影响
    """
    if not isinstance(text, str):
        return text
    return _TELEMETRY_HEADER_RE.sub("", text)


def _strip_in_place(msg: dict) -> None:
    """就地处理单条消息的 content 字段。"""
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = strip_telemetry_from_text(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = strip_telemetry_from_text(block["text"])
    # 其它类型（数字 / None / 自定义对象）静默跳过


def strip_telemetry_from_messages(messages: Iterable[dict] | None) -> None:
    """就地剥离消息列表中的 telemetry header 行。

    扫描范围：
    - 所有 role=system 消息
    - 第一条 role=user 消息（Claude Code 的注入位置）

    其它后续 user 消息可能合法引用 `x-anthropic-*: ...` 字符串作为示例，不处理。
    """
    if not isinstance(messages, list) or not messages:
        return

    try:
        first_user_seen = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                _strip_in_place(msg)
            elif role == "user" and not first_user_seen:
                _strip_in_place(msg)
                first_user_seen = True
    except Exception as e:
        # 永不阻塞主请求：异常仅记日志，messages 保持原样
        logger.warning("strip_telemetry_from_messages 异常，跳过: %s", e)
```

- [ ] **Step 4: 运行测试确认全部通过**

Run: `python -m pytest tests/test_strip_telemetry.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add deep_proxy/optimization/strip_telemetry.py tests/test_strip_telemetry.py
git commit -m "feat(optimization): 新增 strip_telemetry 模块剥离客户端 telemetry header 行

针对 Claude Code 2.1.42+ 在 system prompt 注入 x-anthropic-billing-header
的 session hash，破坏 prefix cache 的问题。共享正则供 compressor 复用，
单一来源避免漂移。"
```

---

### Task 2: compressor 复用共享正则

**Files:**
- Modify: `deep_proxy/optimization/compressor.py:29-35`（删除本地正则 + 改导入）
- Modify: `deep_proxy/optimization/compressor.py:200-210`（`_normalize` 改用 `strip_telemetry_from_text`）

- [ ] **Step 1: 先确认现有 compressor 测试通过（基线）**

Run: `python -m pytest tests/test_compressor.py -v`
Expected: PASS

- [ ] **Step 2: 编辑 compressor.py — 删除本地正则，改 import**

将 `deep_proxy/optimization/compressor.py` 第 29-35 行：

```python
# 动态 header 剥离模式（计算缓存 key 前归一化，避免同一 prompt 因 session ID 不同反复 miss）
# Claude Code 每次会话发送不同的 header（billing 头含 session hash）；同类 CLI 工具
# 也有类似遥测元数据。这些行对下游 LLM 行为无意义，剥离后归一化缓存 key。
_DYNAMIC_HEADERS_RE = re.compile(
    r'^x-anthropic-[a-z-]+:.*$',
    re.MULTILINE | re.IGNORECASE,
)
```

替换为（删除本地定义，改用共享版本）：

```python
# 缓存 key 归一化复用 strip_telemetry 的共享正则——单一来源，避免两处定义漂移。
from .strip_telemetry import strip_telemetry_from_text as _strip_telemetry
```

并将 `_normalize` 方法（`compressor.py:200-210`）改写：

```python
    @staticmethod
    def _normalize(text: str) -> str:
        """在计算缓存 key 前移除会话动态内容。

        当前剥离（均为编码 CLI 遥测元数据，对下游 LLM 语义无影响）：
        - `x-anthropic-*` 系列 header — Claude Code 会话级追踪头（含 session hash）

        与 router.prepare_request 中的 strip_telemetry_from_messages 形成双层防御：
        前者作用于 request body 整体（影响下游传输），本处作用于压缩 cache key。
        共享同一正则（strip_telemetry._TELEMETRY_HEADER_RE）保证语义一致。
        """
        return _strip_telemetry(text)
```

如果文件中 `import re` 没有其它用途（搜索其它 `re.` 用法），保留——`_FENCE_RE` 仍在用，`re` 必须保留。无需改动 import 行。

- [ ] **Step 3: 运行回归测试**

Run: `python -m pytest tests/test_compressor.py tests/test_strip_telemetry.py -v`
Expected: 全部 PASS（compressor 归一化行为不变，因正则相同）

- [ ] **Step 4: 提交**

```bash
git add deep_proxy/optimization/compressor.py
git commit -m "refactor(compressor): 复用 strip_telemetry 共享正则，消除重复定义"
```

---

### Task 3: 配置开关 `strip_client_telemetry`

**Files:**
- Modify: `deep_proxy/config.py:36-40`（OptimizationConfig 主开关附近）

- [ ] **Step 1: 写一个最小测试验证字段默认值**

新建 `tests/test_strip_telemetry_config.py`：

```python
"""验证 OptimizationConfig.strip_client_telemetry 字段默认值与可选关闭。"""
from deep_proxy.config import OptimizationConfig


def test_strip_client_telemetry_default_true():
    cfg = OptimizationConfig()
    assert cfg.strip_client_telemetry is True


def test_strip_client_telemetry_can_be_disabled():
    cfg = OptimizationConfig(strip_client_telemetry=False)
    assert cfg.strip_client_telemetry is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_strip_telemetry_config.py -v`
Expected: FAIL with `AttributeError: 'OptimizationConfig' object has no attribute 'strip_client_telemetry'`

- [ ] **Step 3: 在 `OptimizationConfig` 加字段**

定位 `deep_proxy/config.py` 中 `OptimizationConfig` 的 `enabled` 字段（约 36-37 行）：

```python
    # 主开关
    enabled: bool = Field(default=True, description="是否启用提示词优化")
```

在其紧后面（保持"主开关"分组语义）加入：

```python
    # 客户端 telemetry 剥离（独立于其它 skills 的清理步骤）
    strip_client_telemetry: bool = Field(
        default=True,
        description="剥离客户端在 system / 首条 user 消息中嵌入的 telemetry header 行"
                    "（如 Claude Code 2.1.42+ 的 x-anthropic-billing-header）。"
                    "这些行含 session hash，每次新会话变化，破坏 DeepSeek prefix cache。"
                    "默认开启；与 optimization.enabled 独立——即便其它 skills 全部关闭，"
                    "本步骤仍可单独执行（仅依赖 enabled 主开关与本字段同时为 true）。",
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_strip_telemetry_config.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add deep_proxy/config.py tests/test_strip_telemetry_config.py
git commit -m "feat(config): OptimizationConfig 加 strip_client_telemetry 开关（默认开）"
```

---

### Task 4: `router.prepare_request` 接线

**Files:**
- Modify: `deep_proxy/router.py:139-148` 之间插入 telemetry strip 步骤
- Modify: `deep_proxy/router.py` import 区加入 `strip_telemetry_from_messages`

**位置说明**：
原 spec 写"step 3.5"，实施时为了让 Flash→Pro 升格的 `_upgrade_tracker.snapshot_keys(messages)`
（router.py:148 附近的 `_maybe_upgrade`）也读到干净消息体，**放在 model normalize 后、Flash→Pro 升格前**
（在原 0b 与 0c 之间）。这与 spec 的语义一致（早于所有 skills / 压缩 / 升格读消息体的路径），
只是数字编号微调。

- [ ] **Step 1: 修改 router.py import 段**

定位 `deep_proxy/router.py:43-44`：

```python
from .optimization import apply_cheap_optimizations, extract_cot_output, sample_in_range
from .optimization.compressor import SystemPromptCompressor
```

紧后面加入：

```python
from .optimization.strip_telemetry import strip_telemetry_from_messages
```

- [ ] **Step 2: 在 prepare_request 中插入剥离步骤**

定位 `deep_proxy/router.py:139` 附近（model 归一化之后）：

```python
        # 0b. 模型名称规范化（reasoner/chat 都会被映射到 v4-flash）
        body["model"] = normalize_model_name(raw_model, self._model_routes_dicts)
        model = body.get("model", "")
```

紧后面（在 `# 0c. Flash→Pro 选择性升格路由` 之前）插入：

```python
        # 0c. 客户端 telemetry header 剥离（在升格哈希 / skills / 压缩缓存 key 之前）
        #     Claude Code 2.1.42+ 在 system 头部注入 `x-anthropic-billing-header: cc_version=...`
        #     含 session hash，每次新会话破坏 prefix cache。早期清理让所有下游看到稳定文本。
        #     与 compressor 内部的 _normalize 形成双层防御。
        if (
            self.config.optimization.enabled
            and self.config.optimization.strip_client_telemetry
        ):
            messages = body.get("messages")
            if isinstance(messages, list):
                strip_telemetry_from_messages(messages)
```

随后将原 `# 0c. Flash→Pro 选择性升格路由` 注释序号改为 `# 0d.`，并将后续 `# 1.`、`# 2.` 等
注释序号**不**改动（这些大编号是 CLAUDE.md 文档中的对外 pipeline 阶段，与本步骤是子阶段层级，
不在同一编号体系；本步骤纯粹是 0 系列前置清理）。

- [ ] **Step 3: 运行现有 router pipeline 测试确认不回归**

Run: `python -m pytest tests/test_router_pipeline.py -v`
Expected: PASS（原有测试用例的 messages 不含 telemetry header，新步骤 no-op）

- [ ] **Step 4: 提交**

```bash
git add deep_proxy/router.py
git commit -m "feat(router): prepare_request 早期剥离客户端 telemetry header 行

在 model normalize 后、Flash→Pro 升格判定前调用 strip_telemetry_from_messages，
让升格哈希 / skills / 压缩缓存 key 都读到稳定文本。与 compressor._normalize
形成双层防御。"
```

---

### Task 5: router pipeline 集成测试

**Files:**
- Modify: `tests/test_router_pipeline.py` 文件末尾追加新 class

- [ ] **Step 1: 写测试**

在 `tests/test_router_pipeline.py` 末尾追加：

```python
class TestTelemetryStripping:
    """验证 router.prepare_request 早期 telemetry header 剥离。"""

    async def test_billing_header_stripped_from_system(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": (
                    "x-anthropic-billing-header: cc_version=2.1.42.abc;"
                    " cc_entrypoint=claude-code; cch=0;\n"
                    "You are Claude Code."
                )},
                {"role": "user", "content": "hi"},
            ],
        }
        p = await router.prepare_request(body)
        sys_content = p["messages"][0]["content"]
        assert "x-anthropic-billing-header" not in sys_content
        assert "You are Claude Code." in sys_content

    async def test_first_user_message_also_stripped(self, router: DeepProxyRouter):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": (
                    "x-anthropic-foo: telemetry\nactual question"
                )},
            ],
        }
        p = await router.prepare_request(body)
        assert "x-anthropic-foo" not in p["messages"][0]["content"]
        assert "actual question" in p["messages"][0]["content"]

    async def test_disabled_by_config(self, cfg: ProxyConfig):
        """关闭开关后 header 透传。"""
        cfg.optimization.strip_client_telemetry = False
        local_router = DeepProxyRouter(cfg)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "x-anthropic-foo: keep\nbase"},
                {"role": "user", "content": "hi"},
            ],
        }
        p = await local_router.prepare_request(body)
        assert "x-anthropic-foo: keep" in p["messages"][0]["content"]

    async def test_no_op_when_no_telemetry(self, router: DeepProxyRouter):
        """普通请求无 header 时，messages 内容除其它 skills 注入外保持稳定。"""
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Plain system."},
                {"role": "user", "content": "hello"},
            ],
        }
        p = await router.prepare_request(body)
        # 不抛、不丢失原文（注：其它 skills 可能在 system 头/尾追加，但保留 'Plain system.'）
        assert "Plain system." in p["messages"][0]["content"]
```

如果 `tests/test_router_pipeline.py` 顶部还没有导入 `ProxyConfig`，确保该 import 存在（已在 file head 导入）。

- [ ] **Step 2: 运行**

Run: `python -m pytest tests/test_router_pipeline.py::TestTelemetryStripping -v`
Expected: 全部 PASS

- [ ] **Step 3: 跑全量回归**

Run: `python -m pytest`
Expected: 全部 PASS（无单元 / 集成回归）

- [ ] **Step 4: 提交**

```bash
git add tests/test_router_pipeline.py
git commit -m "test(router): 覆盖 prepare_request 中 telemetry 剥离的开启/关闭路径"
```

---

### Task 6: `config.example.yaml` 更新

**Files:**
- Modify: `config.example.yaml:20-22` 之间

- [ ] **Step 1: 在 `optimization.enabled` 后插入新字段**

定位现有：

```yaml
optimization:
  enabled: true

  # 元功能：LLM 压缩 + 持久化磁盘缓存
```

改为：

```yaml
optimization:
  enabled: true

  # 客户端 telemetry 剥离（在 skills / 压缩 / 升格判定之前执行）
  # 针对 Claude Code 2.1.42+ 在 system prompt 注入的 x-anthropic-billing-header 等
  # 含 session hash 的伪 header 行（破坏 prefix cache）。默认开启。
  strip_client_telemetry: true

  # 元功能：LLM 压缩 + 持久化磁盘缓存
```

- [ ] **Step 2: 确认 yaml 仍可被解析**

Run: `python -c "from deep_proxy.config import ProxyConfig; ProxyConfig.from_yaml('config.example.yaml'); print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 3: 提交**

```bash
git add config.example.yaml
git commit -m "docs(config): config.example.yaml 加 strip_client_telemetry 示例"
```

---

### Task 7: PowerShell 主脚本

**Files:**
- Create: `tools/setup_claude_code_env.ps1`

- [ ] **Step 1: 写脚本**

```powershell
<#
.SYNOPSIS
    一键配置 Windows 用户级永久环境变量，让 Claude Code 接入本地 DeepProxy。

.DESCRIPTION
    1. 读取 config.yaml 拿到 host / coding_port / writing_port / api_key
       （host=0.0.0.0 自动改写为 127.0.0.1；缺失则用默认值）
    2. 用 [Environment]::SetEnvironmentVariable($name, $value, "User") 持久化写入
       用户级环境变量（等价 setx，无需管理员，永久生效，且能写 $null 删除）
    3. 同步写入当前 session（$env:VAR=value），让当前终端立即可用
    4. 主动删除 ANTHROPIC_API_KEY（若存在），避免与 ANTHROPIC_AUTH_TOKEN 冲突
    5. 配置 CLAUDE_CODE_ATTRIBUTION_HEADER=false 从客户端源头关掉 billing header

.PARAMETER DryRun
    只打印将写入的值，不实际写。

.PARAMETER Uninstall
    删除本脚本曾设置的 5 个变量（不动 ANTHROPIC_API_KEY）。

.PARAMETER Writing
    指向 writing_port 而非默认 coding_port。

.PARAMETER Force
    跳过已有 ANTHROPIC_API_KEY 的覆盖确认。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_claude_code_env.ps1
    powershell -ExecutionPolicy Bypass -File setup_claude_code_env.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File setup_claude_code_env.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Uninstall,
    [switch]$Writing,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$MANAGED_VARS = @(
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_MODEL',
    'ANTHROPIC_SMALL_FAST_MODEL',
    'CLAUDE_CODE_ATTRIBUTION_HEADER'
)

function Write-Section($title) {
    Write-Host ""
    Write-Host ("=" * 64) -ForegroundColor DarkGray
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host ("=" * 64) -ForegroundColor DarkGray
}

function Mask-Token([string]$token) {
    if ([string]::IsNullOrEmpty($token)) { return '<empty>' }
    if ($token.Length -le 10) { return ($token.Substring(0, [Math]::Min(2, $token.Length)) + '***') }
    return ($token.Substring(0, 6) + '...' + $token.Substring($token.Length - 2) + " (len=$($token.Length))")
}

function Set-UserEnv([string]$name, [string]$value) {
    # User 级永久写入；同时写入当前 session
    [Environment]::SetEnvironmentVariable($name, $value, 'User')
    if ($null -eq $value) {
        Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
    } else {
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Parse-ConfigYaml([string]$path) {
    # 正则提取顶层 host / coding_port / writing_port / api_key —— 不引入 YAML 模块依赖
    $defaults = @{
        host = '127.0.0.1'
        coding_port = 8000
        writing_port = 8001
        api_key = $null
    }
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Warning "config.yaml 未找到: $path（使用默认值）"
        return $defaults
    }
    try {
        $lines = Get-Content -LiteralPath $path -Encoding UTF8
    } catch {
        Write-Warning "读取 config.yaml 失败: $_（使用默认值）"
        return $defaults
    }
    $result = $defaults.Clone()
    foreach ($line in $lines) {
        # 仅处理顶层字段（行首无缩进），跳过 deepseek: 嵌套层
        if ($line -match '^host:\s*["'']?([^"''\s#]+)') {
            $result.host = $matches[1]
        } elseif ($line -match '^coding_port:\s*(\d+)') {
            $result.coding_port = [int]$matches[1]
        } elseif ($line -match '^writing_port:\s*(\d+)') {
            $result.writing_port = [int]$matches[1]
        } elseif ($line -match '^api_key:\s*["'']?([^"''\s#]+)') {
            $val = $matches[1]
            if ($val -ne 'null') { $result.api_key = $val }
        }
    }
    if ($result.host -eq '0.0.0.0') { $result.host = '127.0.0.1' }
    return $result
}

# -------- 主流程 --------

$scriptDir = Split-Path -LiteralPath $PSCommandPath -Parent
$repoRoot  = Split-Path -LiteralPath $scriptDir -Parent
$configPath = Join-Path $repoRoot 'config.yaml'

if ($Uninstall) {
    Write-Section "Uninstall — 删除 Claude Code 环境变量"
    foreach ($name in $MANAGED_VARS) {
        if ($DryRun) {
            Write-Host "  [DryRun] 将删除 $name" -ForegroundColor Yellow
        } else {
            Set-UserEnv -name $name -value $null
            Write-Host "  - 已删除 $name" -ForegroundColor Green
        }
    }
    Write-Host ""
    Write-Host "ANTHROPIC_API_KEY 未被删除（脚本仅清理自己设置过的 5 个变量）" -ForegroundColor DarkGray
    Write-Host "请打开新终端使删除生效（当前 session 已同步）" -ForegroundColor Cyan
    exit 0
}

$cfg = Parse-ConfigYaml -path $configPath
$port = if ($Writing) { $cfg.writing_port } else { $cfg.coding_port }
$token = if ($cfg.api_key) { $cfg.api_key } else { 'dummy' }
$baseUrl = "http://$($cfg.host):$port"

# 冲突检测：现有 ANTHROPIC_API_KEY
$existingApiKey = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
if ($existingApiKey -and -not $Force) {
    if (-not $Host.UI.RawUI) {
        Write-Error "检测到 ANTHROPIC_API_KEY 已设置；非交互环境请加 -Force 或先手动删除。"
        exit 3
    }
    Write-Host ""
    Write-Host "检测到已有 ANTHROPIC_API_KEY ($(Mask-Token $existingApiKey))" -ForegroundColor Yellow
    Write-Host "Claude Code 在 ANTHROPIC_API_KEY 与 ANTHROPIC_AUTH_TOKEN 同时存在时会优先用前者，"
    Write-Host "导致本脚本设置的 AUTH_TOKEN 不生效。"
    $reply = Read-Host "是否删除 ANTHROPIC_API_KEY ?(y/N)"
    if ($reply -notmatch '^[yY]') {
        Write-Error "已取消（用户拒绝覆盖）。"
        exit 3
    }
}

$plan = [ordered]@{
    'ANTHROPIC_BASE_URL'              = $baseUrl
    'ANTHROPIC_AUTH_TOKEN'            = $token
    'ANTHROPIC_MODEL'                 = 'deepseek-v4-pro[1m]'
    'ANTHROPIC_SMALL_FAST_MODEL'      = 'deepseek-v4-flash'
    'CLAUDE_CODE_ATTRIBUTION_HEADER'  = 'false'
}

Write-Section "Setup — Claude Code → DeepProxy 环境变量"
Write-Host "  配置来源: $configPath"
Write-Host "  目标端口: $port ($(if($Writing){'writing'}else{'coding'}))"
Write-Host ""
Write-Host "  将写入用户级永久变量:"
foreach ($k in $plan.Keys) {
    $v = $plan[$k]
    $display = if ($k -eq 'ANTHROPIC_AUTH_TOKEN') { Mask-Token $v } else { $v }
    Write-Host ("    {0,-32} = {1}" -f $k, $display)
}
Write-Host ""
Write-Host "  将删除（避免与 AUTH_TOKEN 冲突）:"
Write-Host ("    {0,-32} = (deleted)" -f 'ANTHROPIC_API_KEY')

if ($DryRun) {
    Write-Host ""
    Write-Host "[DryRun] 未实际写入任何变量。" -ForegroundColor Yellow
    exit 0
}

try {
    foreach ($k in $plan.Keys) {
        Set-UserEnv -name $k -value $plan[$k]
    }
    Set-UserEnv -name 'ANTHROPIC_API_KEY' -value $null
} catch {
    Write-Error "写入失败: $_"
    exit 2
}

Write-Host ""
Write-Host "完成。" -ForegroundColor Green
Write-Host "  当前终端已同步可用；其它已打开的终端需重启后才会读到新值。" -ForegroundColor Cyan
Write-Host "  使用：1) 启动 DeepProxy (双击 start.bat)  2) 新开终端运行 'claude'" -ForegroundColor Cyan
exit 0
```

- [ ] **Step 2: DryRun 自测**

Run（在 PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_claude_code_env.ps1 -DryRun
```

Expected:
- 打印表格含 5 个 `plan` 变量 + `ANTHROPIC_API_KEY = (deleted)`
- 末尾 `[DryRun] 未实际写入任何变量。`
- 退出码 0

- [ ] **Step 3: 真实写入测试（可选；用户机器上）**

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_claude_code_env.ps1
[Environment]::GetEnvironmentVariable('ANTHROPIC_BASE_URL', 'User')  # 期望 http://127.0.0.1:8000
[Environment]::GetEnvironmentVariable('ANTHROPIC_MODEL', 'User')      # 期望 deepseek-v4-pro[1m]
```

- [ ] **Step 4: Uninstall 测试**

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_claude_code_env.ps1 -Uninstall
[Environment]::GetEnvironmentVariable('ANTHROPIC_BASE_URL', 'User')  # 期望 $null
```

- [ ] **Step 5: 提交**

```bash
git add tools/setup_claude_code_env.ps1
git commit -m "feat(tools): Windows 一键配置 PowerShell 脚本

[Environment]::SetEnvironmentVariable User scope 永久写入，无需管理员。
支持 -DryRun / -Uninstall / -Writing / -Force。
主动删除 ANTHROPIC_API_KEY 避免与 AUTH_TOKEN 冲突。"
```

---

### Task 8: `.bat` 启动器

**Files:**
- Create: `tools/setup_claude_code_env.bat`

- [ ] **Step 1: 写启动器**

```batch
@echo off
REM 双击启动 Claude Code 环境变量配置脚本。
REM 透传所有参数到 PowerShell 脚本（-DryRun / -Uninstall / -Writing / -Force）。
REM Usage:
REM   setup_claude_code_env.bat
REM   setup_claude_code_env.bat -DryRun
REM   setup_claude_code_env.bat -Uninstall

setlocal
cd /d "%~dp0"

REM 优先 PowerShell 7 (pwsh)，回退到 Windows 内置 PowerShell 5.1+
where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PSEXE=pwsh"
) else (
    set "PSEXE=powershell"
)

"%PSEXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_claude_code_env.ps1" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] 配置脚本返回非零退出码: %RC%
    pause
)

endlocal & exit /b %RC%
```

- [ ] **Step 2: 自测**

双击 `tools\setup_claude_code_env.bat` 或在 cmd 内：

```cmd
tools\setup_claude_code_env.bat -DryRun
```

Expected: 与 Task 7 Step 2 同样输出，退出码 0。

- [ ] **Step 3: 提交**

```bash
git add tools/setup_claude_code_env.bat
git commit -m "feat(tools): .bat 启动器调用 PowerShell 配置脚本

优先 pwsh (PS7+)，回退到 Windows 5.1+ 自带 powershell。
非零退出码 pause 让双击时错误可见。"
```

---

### Task 9: CLAUDE.md + README.md 文档

**Files:**
- Modify: `CLAUDE.md`（Request Pipeline 段，约在 "Request Pipeline" 章节）
- Modify: `README.md`

- [ ] **Step 1: CLAUDE.md — Request Pipeline 列表加入新步骤**

定位 `CLAUDE.md` 中 "## Request Pipeline" 段，将现有 1-9 步列表的第 1 步之前插入：

```markdown
0. **客户端 telemetry header 剥离** — 早期清理 `system` 与首条 `user` 消息中的 `^x-anthropic-[a-z-]+:.*$` 行（Claude Code 2.1.42+ 的 billing header 等含 session hash，会破坏 prefix cache）。受 `optimization.strip_client_telemetry` 控制，默认开启。
```

并把原有 1-9 重新编号变为 1-10（编号顺移）。

- [ ] **Step 2: CLAUDE.md — 项目结构段加新文件**

定位 `## Project Structure` 中 `optimization/` 子目录列表，加：

```
│   │   ├── strip_telemetry.py     # x-anthropic-* 行剥离（共享正则供 compressor / router 双层防御复用）
```

`tools/` 子目录加：

```
│   ├── setup_claude_code_env.ps1 # Windows 一键配置 Claude Code 环境变量（用户级永久）
│   └── setup_claude_code_env.bat # 双击启动器
```

- [ ] **Step 3: README.md — 加 Windows 一键接入段**

在 README.md 现有 "## 启动 / 安装" 或相近段落之后加：

```markdown
## Windows 一键接入 Claude Code

DeepProxy 暴露 Anthropic 兼容的 `/v1/messages` 端点。在 Windows 上一键配置 Claude Code
所有相关环境变量（用户级永久、无需管理员）：

1. 启动 DeepProxy（`start.bat`）
2. 双击 `tools\setup_claude_code_env.bat`（或在终端：`tools\setup_claude_code_env.bat`）
3. 打开**新终端**运行 `claude`

脚本会读取 `config.yaml` 的 `host` / `coding_port` / `api_key`，写入以下用户级永久环境变量：

| 变量 | 默认值 |
|------|--------|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8000` |
| `ANTHROPIC_AUTH_TOKEN` | 来自 `config.yaml` 的 `api_key`（缺省 `dummy`） |
| `ANTHROPIC_MODEL` | `deepseek-v4-pro[1m]` |
| `ANTHROPIC_SMALL_FAST_MODEL` | `deepseek-v4-flash` |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `false`（从源头关掉 billing header；代理层也独立剥离作为兜底） |

同时主动删除 `ANTHROPIC_API_KEY` 以避免与 `AUTH_TOKEN` 优先级冲突（已设置时会先确认）。

**参数**：

- `-DryRun`：只打印不写
- `-Uninstall`：删除上述 5 个变量
- `-Writing`：指向 `writing_port`（默认 coding）
- `-Force`：跳过 `ANTHROPIC_API_KEY` 覆盖确认（CI 场景）

举例：

```cmd
tools\setup_claude_code_env.bat -DryRun
tools\setup_claude_code_env.bat -Uninstall
```
```

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md README.md
git commit -m "docs: CLAUDE.md / README.md 记录 telemetry 剥离与 Windows 一键接入脚本"
```

---

## Self-Review

**1. Spec coverage**

| Spec 章节 | 实施 Task |
|-----------|-----------|
| §3.1.1–§3.1.2 PowerShell 主脚本 | Task 7 |
| §3.1.3 用 SetEnvironmentVariable / 不引 admin / 当前 session 同步 / mask token | Task 7 Step 1（`Set-UserEnv` / `Mask-Token`） |
| §3.1.4 YAML 正则解析（不引依赖） | Task 7 Step 1（`Parse-ConfigYaml`） |
| §3.2.1 共享正则 + 函数 | Task 1 |
| §3.2.2 compressor 复用 | Task 2 |
| §3.2.3 router pipeline 接线 | Task 4 |
| §3.2.4 配置开关 `strip_client_telemetry` | Task 3 |
| §4 错误处理（脚本：DryRun / Uninstall / Force / IO 异常；过滤：容错优先） | Tasks 7（脚本分支）+ 1（`strip_telemetry_from_messages` try/except） |
| §5 单元 / 集成 / 回归测试 | Tasks 1（unit）+ 5（integration）+ 2（compressor 回归） |
| §6 文档更新（README / CLAUDE.md / config.example.yaml） | Tasks 6 + 9 |
| §7 风险（仅 x-anthropic-* 前缀） | Task 1 注释中明示，正则按设计 |
| §9 验收标准 | Task 7/8 的 DryRun + 真实写入 + Uninstall 自测覆盖 1-3、Task 5 集成测试覆盖 4-5、Task 7/8 覆盖 6 |

**2. Placeholder scan** — 无 TBD / TODO / "implement later"。所有代码块完整。Tasks 间引用具体方法名一致（`strip_telemetry_from_text` / `strip_telemetry_from_messages` / `_TELEMETRY_HEADER_RE` / `strip_client_telemetry`）。

**3. Type / 名称一致性**

- `_TELEMETRY_HEADER_RE`（不是 `_DYNAMIC_HEADERS_RE`）— Task 1 定义，Task 2 复用，文档/spec 一致
- `strip_telemetry_from_messages`（in-place / void return）— Task 1 定义，Task 4 调用方式匹配
- `strip_client_telemetry`（配置字段名）— Task 3 定义，Task 4 + Task 5 + Task 6 + Task 9 引用一致
- `MANAGED_VARS`（5 个变量）— Task 7 与 Task 8 与 Task 9 README 表格一致（不含 `ANTHROPIC_API_KEY`）

无名称漂移。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-19-claude-code-env-setup.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 task 派一个新 subagent，跨 task review，快速迭代
**2. Inline Execution** — 在当前 session 顺序执行，每个 task 完成后 checkpoint review

Which approach?
