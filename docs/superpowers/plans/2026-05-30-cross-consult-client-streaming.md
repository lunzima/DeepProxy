# Cross-Consult 客户端真流式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cross_consult 激活时，让流式 endpoint `iter_chat_chunks` 对客户端真流式逐 token 推送（content/reasoning 透传、抑制 cc 工具帧、consult 间隙发 keep-alive 心跳、跨重发轮桥接），取代当前「buffer 一切再合成单块」的假流式。

**Architecture:** 新增 `deep_proxy/cross_consult/client_stream.py`，含三个隔离单元：`stream_one_turn`（单轮流式器，content/reasoning 即时透传、tool_calls 累加到轮末判定、间隙发心跳）、`with_heartbeat`（包裹 consult await 期间发心跳）、`stream_cross_consult_continuation`（`execute_cross_consult_loop` 的流式变体）。`iter_chat_chunks` 改为透传 + 交棒给 continuation；协议层 `chat_completions_stream` 把心跳 sentinel 序列化成 SSE 注释帧。非流式 endpoint 不变。

**Tech Stack:** Python 3.12+ / asyncio / pytest（`asyncio_mode=auto`）/ FastAPI / LiteLLM。

---

## 设计参考

实现前阅读 spec：`docs/superpowers/specs/2026-05-30-cross-consult-client-streaming-design.md`。

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `deep_proxy/cross_consult/client_stream.py` | 三单元 + TurnResult/_Done | 新建 |
| `deep_proxy/cross_consult/config.py` | `stream_heartbeat_seconds` 字段 | 改 |
| `deep_proxy/router.py` | `iter_chat_chunks` 接线 + `chat_completions_stream` 心跳序列化 + 清理死代码 | 改 |
| `deep_proxy/cross_consult/interceptor.py` | 删除死掉的 `synthesize_final_stream_chunk` | 改 |
| `config.example.yaml` | 心跳配置注释 | 改 |
| `tests/test_cross_consult_client_stream.py` | 三单元单测 | 新建 |
| `tests/test_cross_consult_loop.py` | 集成测试改写/新增 | 改 |
| `tests/test_cross_consult_config.py` | 心跳配置默认值断言 | 改 |

## 复用的现有符号（不要重新实现）

- `merge_tool_call_deltas(existing, new) -> list` — `deep_proxy/utils.py`（name 覆盖、arguments 拼接）
- `SSE_DONE = "data: [DONE]\n\n"` — `deep_proxy/utils.py:362`
- `iter_litellm_chunks(config, body, *, _accumulator=None, provider=None)` — `deep_proxy/litellm_client.py:254`（流式 dict 产出，已做 reasoning 规整 + null 清理）
- `StreamingReasoningAccumulator(request_messages=...)` — `deep_proxy/compatibility/reasoning_handler.py`
- `_extract_cross_consult_tool_calls(response, tool_name) -> list[dict]` — `deep_proxy/cross_consult/interceptor.py:88`
- `_resolve_consult_tool_call(tc, *, call_count, target_provider, config, cc_config) -> tuple[str, bool]` — `interceptor.py:161`（返回 `(tool_text, consumed_quota)`）
- `build_initial_response_from_stream_tool_calls(tool_calls) -> dict` — `interceptor.py:104`（包成 `{"choices":[{"message":{...,"tool_calls":tcs},"finish_reason":"tool_calls"}]}`）

---

## Task 1: 心跳配置 `stream_heartbeat_seconds`

**Files:**
- Modify: `deep_proxy/cross_consult/config.py`（在 `first_chunk_timeout_seconds` 字段之后）
- Modify: `config.example.yaml`（cross_consult 块）
- Test: `tests/test_cross_consult_config.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cross_consult_config.py` 末尾追加：

```python
def test_cross_consult_config_stream_heartbeat_default():
    from deep_proxy.cross_consult.config import CrossConsultConfig
    cc = CrossConsultConfig()
    assert cc.stream_heartbeat_seconds == 10


def test_cross_consult_config_stream_heartbeat_bounds():
    import pytest
    from pydantic import ValidationError
    from deep_proxy.cross_consult.config import CrossConsultConfig
    with pytest.raises(ValidationError):
        CrossConsultConfig(stream_heartbeat_seconds=0)
    with pytest.raises(ValidationError):
        CrossConsultConfig(stream_heartbeat_seconds=121)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_config.py::test_cross_consult_config_stream_heartbeat_default -v`
Expected: FAIL — `AttributeError`/断言失败（字段不存在）。

- [ ] **Step 3: Add the config field**

在 `deep_proxy/cross_consult/config.py` 的 `first_chunk_timeout_seconds` 字段块之后插入：

```python
    stream_heartbeat_seconds: int = Field(
        default=10, ge=1, le=120,
        description="客户端真流式下，静默间隙（consult 执行 / 重发 prefill）期间发送 "
                    "SSE keep-alive 注释帧的间隔秒数。须显著小于客户端 idle-read 超时。",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_consult_config.py -v`
Expected: PASS（含两个新测试）。

- [ ] **Step 5: Document in example config**

在 `config.example.yaml` 的 `first_chunk_timeout_seconds:` 行之后插入：

```yaml
  stream_heartbeat_seconds: 10          # 客户端真流式静默间隙（consult/重发prefill）期发 SSE keep-alive 注释帧的间隔秒
```

- [ ] **Step 6: Commit**

```bash
git add deep_proxy/cross_consult/config.py config.example.yaml tests/test_cross_consult_config.py
git commit -m "feat(cross-consult): 新增 stream_heartbeat_seconds 配置"
```

---

## Task 2: 协议层心跳序列化

`chat_completions_stream`（`router.py:589`）把心跳 sentinel `{"_dp_heartbeat": True}` 序列化成 SSE 注释帧 `: keep-alive\n\n`，普通 dict 仍走 `data: {json}\n\n`。

**Files:**
- Modify: `deep_proxy/router.py:589-601`（`chat_completions_stream`）
- Test: `tests/test_cross_consult_loop.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cross_consult_loop.py` 末尾追加（注意：该文件已 `from unittest.mock import AsyncMock`；如缺 `patch` 按需补 import）：

```python
async def test_chat_completions_stream_serializes_heartbeat_as_sse_comment():
    """心跳 sentinel -> SSE 注释帧（: keep-alive），不是 data: 帧。"""
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    from deep_proxy.config import ProxyConfig, normalize_legacy_config

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    router = DeepProxyRouter(cfg)

    async def fake_iter(body, *, provider=None):
        yield {"_dp_heartbeat": True}
        yield {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}

    with patch.object(router, "iter_chat_chunks", new=fake_iter):
        out = [s async for s in router.chat_completions_stream({}, provider=None)]

    assert ": keep-alive\n\n" in out
    # 心跳不得被 json dump 成 data 帧
    assert not any(s.startswith("data: ") and "_dp_heartbeat" in s for s in out)
    # 普通帧仍是 data 帧
    assert any(s.startswith("data: ") and '"content": "hi"' in s for s in out)
    assert out[-1] == "data: [DONE]\n\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_loop.py::test_chat_completions_stream_serializes_heartbeat_as_sse_comment -v`
Expected: FAIL — 心跳被当普通 dict `data: {"_dp_heartbeat": true}` 输出，断言 `": keep-alive\n\n" in out` 失败。

- [ ] **Step 3: Implement serialization**

把 `router.py` 的 `chat_completions_stream` 循环体改为：

```python
        async for item in self.iter_chat_chunks(body, provider=provider):
            if item.get("_dp_heartbeat"):
                # SSE 注释帧：规范明确忽略 `:` 开头行，零风险污染 delta 解析
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if isinstance(item.get("error"), dict) and not item.get("choices"):
                yield SSE_DONE
                return
        yield SSE_DONE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cross_consult_loop.py::test_chat_completions_stream_serializes_heartbeat_as_sse_comment -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/router.py tests/test_cross_consult_loop.py
git commit -m "feat(cross-consult): 协议层把心跳 sentinel 序列化为 SSE 注释帧"
```

---

## Task 3: `with_heartbeat` 心跳包裹

**Files:**
- Create: `deep_proxy/cross_consult/client_stream.py`
- Test: `tests/test_cross_consult_client_stream.py`

- [ ] **Step 1: Write the failing test**

新建 `tests/test_cross_consult_client_stream.py`：

```python
"""client_stream 三单元单测（mock chunk 流 / awaitable，快）。"""
from __future__ import annotations

import asyncio

from deep_proxy.cross_consult.client_stream import with_heartbeat, _Done


async def test_with_heartbeat_emits_pings_then_result():
    async def slow():
        await asyncio.sleep(0.25)
        return "answer"

    frames = []
    result = None
    async for f in with_heartbeat(slow(), heartbeat_seconds=0.1):
        if isinstance(f, _Done):
            result = f.value
        else:
            frames.append(f)

    assert result == "answer"
    assert frames and all(fr == {"_dp_heartbeat": True} for fr in frames)
    assert len(frames) >= 2  # ~0.25s / 0.1s


async def test_with_heartbeat_fast_awaitable_no_ping():
    async def fast():
        return 42

    frames = [f async for f in with_heartbeat(fast(), heartbeat_seconds=5.0)]
    assert len(frames) == 1
    assert isinstance(frames[0], _Done) and frames[0].value == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -v`
Expected: FAIL — `ModuleNotFoundError: deep_proxy.cross_consult.client_stream`。

- [ ] **Step 3: Create client_stream.py with `_Done` + `with_heartbeat`**

新建 `deep_proxy/cross_consult/client_stream.py`：

```python
"""客户端真流式：cross_consult 激活时逐 token 透传 + 抑制虚拟工具帧 + 心跳桥接。

三单元：
  - with_heartbeat：包裹 consult await，期间周期 yield 心跳帧
  - stream_one_turn：消费单轮上游 chunk 流，content/reasoning 即时透传、
    tool_calls 累加到轮末判定、间隙发心跳
  - stream_cross_consult_continuation：execute_cross_consult_loop 的流式变体

心跳 sentinel = {"_dp_heartbeat": True}（dict），由协议层序列化成 SSE 注释帧。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Generic, TypeVar

logger = logging.getLogger(__name__)

_HEARTBEAT: dict[str, Any] = {"_dp_heartbeat": True}

T = TypeVar("T")


@dataclass
class _Done(Generic[T]):
    """with_heartbeat 的终结哨兵：携带被包裹 awaitable 的结果。"""
    value: T


async def with_heartbeat(
    awaitable: Awaitable[T], *, heartbeat_seconds: float,
) -> AsyncGenerator[Any, None]:
    """运行 awaitable，期间每 heartbeat_seconds 无完成就 yield 一个心跳帧；
    完成后 yield 单个 _Done(result)。"""
    task = asyncio.ensure_future(awaitable)
    while True:
        done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
        if task in done:
            yield _Done(task.result())
            return
        yield _HEARTBEAT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -v`
Expected: PASS（2 项）。

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_cross_consult_client_stream.py
git commit -m "feat(cross-consult): client_stream with_heartbeat 单元"
```

---

## Task 4: `stream_one_turn` 单轮流式器

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py`（追加 `TurnResult`、`_client_facing_chunk`、`stream_one_turn`）
- Test: `tests/test_cross_consult_client_stream.py`

- [ ] **Step 1: Write the failing tests**

在 `tests/test_cross_consult_client_stream.py` 追加：

```python
from deep_proxy.cross_consult.client_stream import stream_one_turn, TurnResult


def _delta_chunk(**delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _finish_chunk(reason):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def _cc_tool_call_delta():
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "cc1", "type": "function",
         "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}
    ]}, "finish_reason": None}]}


async def _iter(chunks):
    for c in chunks:
        yield c


async def test_stream_one_turn_forwards_content_and_reasoning_live():
    chunks = [
        _delta_chunk(role="assistant"),
        _delta_chunk(reasoning_content="思考"),
        _delta_chunk(content="答案"),
        _finish_chunk("stop"),
    ]
    res = TurnResult()
    out = [f async for f in stream_one_turn(
        _iter(chunks), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    texts = [d["choices"][0]["delta"] for d in out if "choices" in d]
    assert {"reasoning_content": "思考"} in texts
    assert {"content": "答案"} in texts
    assert res.had_cc_call is False
    assert res.content == "答案"
    assert res.finish_reason == "stop"


async def test_stream_one_turn_suppresses_cc_tool_call_frames():
    chunks = [
        _delta_chunk(content="让我咨询"),
        _cc_tool_call_delta(),
        _finish_chunk("tool_calls"),
    ]
    res = TurnResult()
    out = [f async for f in stream_one_turn(
        _iter(chunks), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    # 前导 content 透传
    assert any(d.get("choices", [{}])[0].get("delta") == {"content": "让我咨询"} for d in out)
    # cc 工具帧不透传；finish_reason=tool_calls 不透传
    assert not any("tool_calls" in d.get("choices", [{}])[0].get("delta", {}) for d in out)
    assert not any(d.get("choices", [{}])[0].get("finish_reason") for d in out)
    # 但 cc 调用被累加 + 标记
    assert res.had_cc_call is True
    assert res.accumulated_tool_calls
    assert res.accumulated_tool_calls[0]["function"]["name"] == "cross_consult"


async def test_stream_one_turn_emits_heartbeat_on_gap():
    async def slow_gen():
        yield _delta_chunk(content="a")
        await asyncio.sleep(0.25)
        yield _delta_chunk(content="b")
        yield _finish_chunk("stop")

    res = TurnResult()
    out = [f async for f in stream_one_turn(
        slow_gen(), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert any(f == {"_dp_heartbeat": True} for f in out)
    assert res.errored is False


async def test_stream_one_turn_errors_when_budget_exceeded():
    async def hang_gen():
        await asyncio.sleep(1.0)
        yield _delta_chunk(content="never")

    res = TurnResult()
    out = [f async for f in stream_one_turn(
        hang_gen(), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=0.2, heartbeat_seconds=0.1,
    )]
    assert res.errored is True
    # 预算内仍发了心跳
    assert any(f == {"_dp_heartbeat": True} for f in out)


async def test_stream_one_turn_forwards_error_frame():
    chunks = [_delta_chunk(content="x"), {"error": {"message": "boom"}}]
    res = TurnResult()
    out = [f async for f in stream_one_turn(
        _iter(chunks), res, tool_name="cross_consult",
        idle_timeout=5.0, first_chunk_timeout=5.0, heartbeat_seconds=5.0,
    )]
    assert any(d.get("error") for d in out)
    assert res.errored is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -k stream_one_turn -v`
Expected: FAIL — `ImportError`（`stream_one_turn` / `TurnResult` 未定义）。

- [ ] **Step 3: Implement `TurnResult`, `_client_facing_chunk`, `stream_one_turn`**

在 `client_stream.py` 追加（顶部 import 已含 dataclass/field/asyncio）：

```python
from ..utils import merge_tool_call_deltas


@dataclass
class TurnResult:
    accumulated_tool_calls: list[dict] = field(default_factory=list)
    content: str = ""            # 累加的 assistant 文本，供重发轮重建消息历史
    had_cc_call: bool = False
    finish_reason: str | None = None
    errored: bool = False


def _client_facing_chunk(chunk: dict) -> dict | None:
    """从上游 chunk 构造仅含 content/reasoning 的客户端帧（剥 tool_calls、
    抑制 finish_reason）。无可透传内容时返回 None。"""
    out_choices = []
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        fwd: dict[str, Any] = {}
        if delta.get("role"):
            fwd["role"] = delta["role"]
        if isinstance(delta.get("content"), str):
            fwd["content"] = delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            fwd["reasoning_content"] = delta["reasoning_content"]
        if isinstance(delta.get("reasoning"), str):
            fwd["reasoning"] = delta["reasoning"]
        # 仅 role（无 content/reasoning）的空壳不值得单独发
        if not fwd or set(fwd.keys()) == {"role"}:
            continue
        out_choices.append({"index": ch.get("index", 0), "delta": fwd,
                            "finish_reason": None})
    if not out_choices:
        return None
    return {"choices": out_choices}


def _accumulate_turn(chunk: dict, result: TurnResult, tool_name: str) -> None:
    """把一个 chunk 的 tool_calls / content / finish_reason 累加进 result。"""
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        if isinstance(delta.get("content"), str):
            result.content += delta["content"]
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            result.accumulated_tool_calls = merge_tool_call_deltas(
                result.accumulated_tool_calls, tcs,
            )
        fr = ch.get("finish_reason")
        if fr:
            result.finish_reason = fr
    result.had_cc_call = any(
        (tc.get("function") or {}).get("name") == tool_name
        for tc in result.accumulated_tool_calls
    )


async def stream_one_turn(
    chunk_iter: AsyncIterator[dict],
    result: TurnResult,
    *,
    tool_name: str,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
) -> AsyncGenerator[Any, None]:
    """消费单轮上游 chunk 流：content/reasoning 即时透传；tool_calls 累加（不透传）
    留到轮末判定；等待间隙发心跳；error frame / 超预算 -> result.errored=True 并终止。

    心跳/预算：等待下一 chunk 时每 heartbeat_seconds 无 chunk 发一次心跳；累计等待
    超过预算（首 chunk 用 first_chunk_timeout，之后 idle_timeout）视为 hang。
    """
    it = chunk_iter.__aiter__() if hasattr(chunk_iter, "__aiter__") else chunk_iter
    got_first = False
    task: asyncio.Future | None = asyncio.ensure_future(it.__anext__())
    waited = 0.0
    while True:
        budget = idle_timeout if got_first else first_chunk_timeout
        step = heartbeat_seconds
        if budget and budget > 0:
            step = min(heartbeat_seconds, max(0.0, budget - waited))
        done, _ = await asyncio.wait({task}, timeout=step if step > 0 else heartbeat_seconds)
        if task not in done:
            waited += step
            if budget and budget > 0 and waited >= budget:
                logger.warning(
                    "stream_one_turn %s timeout after %.1fs",
                    "first_chunk" if not got_first else "mid_stream", budget,
                )
                result.errored = True
                task.cancel()
                return
            yield _HEARTBEAT
            continue
        # chunk 到达
        try:
            chunk = task.result()
        except StopAsyncIteration:
            return
        got_first = True
        waited = 0.0
        task = asyncio.ensure_future(it.__anext__())

        if isinstance(chunk.get("error"), dict) and not chunk.get("choices"):
            result.errored = True
            yield chunk
            task.cancel()
            return

        _accumulate_turn(chunk, result, tool_name)
        fwd = _client_facing_chunk(chunk)
        if fwd is not None:
            yield fwd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -v`
Expected: PASS（with_heartbeat 2 项 + stream_one_turn 5 项）。

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_cross_consult_client_stream.py
git commit -m "feat(cross-consult): client_stream stream_one_turn 单元"
```

---

## Task 5: `stream_cross_consult_continuation` 流式 continuation

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py`（追加 continuation）
- Test: `tests/test_cross_consult_client_stream.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cross_consult_client_stream.py` 追加：

```python
from unittest.mock import patch

from deep_proxy.cross_consult.client_stream import stream_cross_consult_continuation
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.config import ProxyConfig, normalize_legacy_config
from deep_proxy.providers import Provider
from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator


def _cc_tool_call(call_id="cc1", question="q"):
    return {"index": 0, "id": call_id, "type": "function",
            "function": {"name": "cross_consult",
                         "arguments": '{"question":"%s"}' % question}}


async def test_continuation_streams_consult_heartbeat_and_resend():
    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"},
    }))
    cfg.cross_consult = CrossConsultConfig(
        enabled=True, pairs={"deepseek": "mimo", "mimo": "deepseek"},
    )
    cfg.providers["mimo"] = Provider(
        name="mimo", api_base="https://x", api_key="t", litellm_prefix="openai/",
        flash_model="mimo-v2.5", pro_model="mimo-v2.5-pro",
    )
    source = cfg.providers["deepseek"]
    acc = StreamingReasoningAccumulator(request_messages=[])

    # 重发轮的流式 chunk（无 cc 调用 -> 终轮）
    async def resend_iter(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "综合答案"},
                            "finish_reason": "stop"}]}

    body = {"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "use cc"}]}

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks",
               new=resend_iter):
        frames = [f async for f in stream_cross_consult_continuation(
            initial_tool_calls=[_cc_tool_call()],
            body=body, source_provider=source, config=cfg,
            cc_config=cfg.cross_consult, accumulator=acc,
        )]

    # 重发 content 逐帧透传
    assert any(fr.get("choices", [{}])[0].get("delta", {}).get("content") == "综合答案"
               for fr in frames)
    # tool_result 已注入消息历史
    assert any(m.get("role") == "tool" and "外部视角" in str(m.get("content"))
               for m in body["messages"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_client_stream.py::test_continuation_streams_consult_heartbeat_and_resend -v`
Expected: FAIL — `ImportError`（`stream_cross_consult_continuation` 未定义）。

- [ ] **Step 3: Implement the continuation**

在 `client_stream.py` 追加 import 与函数：

```python
from ..config import ProxyConfig
from ..litellm_client import iter_litellm_chunks
from ..providers import Provider
from .config import CrossConsultConfig
from .interceptor import (
    _extract_cross_consult_tool_calls,
    _resolve_consult_tool_call,
    build_initial_response_from_stream_tool_calls,
)
```

> 注：consult 执行通过 `_resolve_consult_tool_call` 间接进行（它内部 `await execute_consult`）；client_stream **不**直接 import `execute_consult`。因此 mock consult 时须 patch `deep_proxy.cross_consult.interceptor.execute_consult`（`_resolve_consult_tool_call` 所在命名空间），而非 client_stream。

```python
async def stream_cross_consult_continuation(
    *,
    initial_tool_calls: list[dict],
    body: dict[str, Any],
    source_provider: Provider,
    config: ProxyConfig,
    cc_config: CrossConsultConfig,
    accumulator: Any,
) -> AsyncGenerator[Any, None]:
    """execute_cross_consult_loop 的流式变体：执行 consult（间隙发心跳）+ 重发
    （逐 chunk 透传）+ 跨轮循环。yield 客户端帧 / 心跳帧 / error 帧。

    initial_tool_calls：初始轮已累加的 tool_calls（含至少一个 cross_consult 调用）。
    """
    target_name = cc_config.pair_for(source_provider.name)
    target_provider = config.providers.get(target_name) if target_name else None
    if target_provider is None:
        return  # 无对偶，无可继续（调用方已透传初始内容）

    idle = float(cc_config.call_timeout_seconds)
    first = float(cc_config.first_chunk_timeout_seconds)
    hb = float(cc_config.stream_heartbeat_seconds)

    turn_tool_calls = initial_tool_calls
    turn_content = ""
    call_count = 0
    max_turns = cc_config.max_calls_per_request * 2 + 1

    for _turn in range(max_turns):
        pseudo = build_initial_response_from_stream_tool_calls(turn_tool_calls)
        # 终轮判定：本轮无 cross_consult 调用 -> 调用方已/将透传，停止
        cc_calls = _extract_cross_consult_tool_calls(pseudo, cc_config.tool_name)
        if not cc_calls:
            return

        # 追加 assistant 消息（含本轮 content + 全部 tool_calls）到历史
        body["messages"].append({
            "role": "assistant",
            "content": turn_content or None,
            "tool_calls": turn_tool_calls,
        })

        for tc in cc_calls:
            tool_text = None
            async for frame in with_heartbeat(
                _resolve_consult_tool_call(
                    tc, call_count=call_count,
                    target_provider=target_provider, config=config, cc_config=cc_config,
                ),
                heartbeat_seconds=hb,
            ):
                if isinstance(frame, _Done):
                    tool_text, consumed = frame.value
                    if consumed:
                        call_count += 1
                else:
                    yield frame  # 心跳
            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": tool_text,
            })

        # 重发：流式，逐 chunk 透传；复用同一 accumulator 写缓存
        resend_iter = iter_litellm_chunks(
            config, body, _accumulator=accumulator, provider=source_provider,
        )
        turn = TurnResult()
        async for frame in stream_one_turn(
            resend_iter, turn, tool_name=cc_config.tool_name,
            idle_timeout=idle, first_chunk_timeout=first, heartbeat_seconds=hb,
        ):
            yield frame
        if turn.errored:
            return
        turn_tool_calls = turn.accumulated_tool_calls
        turn_content = turn.content
        # 终轮（无 cc 调用）：把本轮 finish_reason / 非 cc tool_calls 作为终结帧透传
        if not turn.had_cc_call:
            final_delta: dict[str, Any] = {}
            if turn_tool_calls:
                final_delta["tool_calls"] = turn_tool_calls
            yield {"choices": [{
                "index": 0,
                "delta": final_delta,
                "finish_reason": turn.finish_reason or "stop",
            }]}
            return

    logger.warning("cross_consult stream continuation hit hard turn limit (%d)", max_turns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -v`
Expected: PASS（全部）。

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_cross_consult_client_stream.py
git commit -m "feat(cross-consult): client_stream 流式 continuation 单元"
```

---

## Task 6: 接线 `iter_chat_chunks` 改为真流式

把 `iter_chat_chunks` 的 `cc_active` 分支从「buffer 一切」改为「stream_one_turn 透传初始流 + 交棒 continuation」。

**Files:**
- Modify: `deep_proxy/router.py`（imports + `iter_chat_chunks` 主体，约 `494-587`）
- Test: `tests/test_cross_consult_loop.py`

- [ ] **Step 1: Write/adapt the failing integration test**

阅读现有 `tests/test_cross_consult_loop.py::test_iter_chat_chunks_intercepts_cross_consult_in_stream`，把它替换为下面这版（断言「逐帧到达」而非「单一合成块」）；并新增前导/多 reasoning/心跳/无cc 四个断言测试。需要文件顶部辅助 `_make_chunk_sequence_iter` / `_text_chunks`（已存在，复用）。

```python
async def test_iter_chat_chunks_streams_cross_consult_live(cfg_cross):
    """cc 激活时，初始 content/reasoning + 重发 content 逐帧到达客户端，
    cc 工具帧不可见。"""
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter

    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    # 初始流：前导 content + cc 工具调用
    async def initial_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"reasoning_content": "想一下"},
                            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"content": "让我咨询"},
                            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "cc1", "type": "function",
             "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}]},
            "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

    # 重发流：终轮 content
    async def resend_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "综合答案"},
                            "finish_reason": "stop"}]}

    # iter_litellm_chunks 被调用两次：初始（router 层）+ 重发（continuation 层）
    calls = {"n": 0}

    def dispatch(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        return initial_stream(config, body) if calls["n"] == 1 else resend_stream(config, body)

    async def consult_ok(**kw):
        return "外部视角"

    with patch("deep_proxy.router.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=consult_ok):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "use cc"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]

    deltas = [fr.get("choices", [{}])[0].get("delta", {}) for fr in frames if "choices" in fr]
    # 初始 reasoning + 前导 content 透传
    assert {"reasoning_content": "想一下"} in deltas
    assert {"content": "让我咨询"} in deltas
    # 重发 content 透传
    assert {"content": "综合答案"} in deltas
    # cc 工具帧不可见
    assert not any("tool_calls" in d and any(
        (tc.get("function") or {}).get("name") == "cross_consult" for tc in d["tool_calls"]
    ) for d in deltas)
```

> 删除旧的 `test_iter_chat_chunks_intercepts_cross_consult_in_stream`（它断言单一合成块，已不成立）。

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_loop.py::test_iter_chat_chunks_streams_cross_consult_live -v`
Expected: FAIL — 当前实现 buffer 一切并 yield 合成单块，逐帧 delta 断言不成立。

- [ ] **Step 3: Add imports to router.py**

`router.py` 的 cross_consult import 块（`55-61`）改为：

```python
from .cross_consult import RedirectTracker
from .cross_consult.interceptor import (
    execute_cross_consult_loop,
    inject_into_request,
)
from .cross_consult.streaming import stream_aggregated_call
from .cross_consult.client_stream import (
    TurnResult,
    stream_cross_consult_continuation,
    stream_one_turn,
)
```

> 注：移除了 `build_initial_response_from_stream_tool_calls` 与 `synthesize_final_stream_chunk` 的 import（前者仅 continuation 内部用；后者将在 Task 7 删除）。

- [ ] **Step 4: Rewrite the `iter_chat_chunks` cc_active body**

把 `iter_chat_chunks` 的 `try:` 主体（`523-582`）整体替换为：

```python
        try:
            if cc_active:
                turn = TurnResult()
                initial_iter = iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                )
                idle = float(self.config.cross_consult.call_timeout_seconds)
                first = float(self.config.cross_consult.first_chunk_timeout_seconds)
                hb = float(self.config.cross_consult.stream_heartbeat_seconds)
                async for frame in stream_one_turn(
                    initial_iter, turn, tool_name=self.config.cross_consult.tool_name,
                    idle_timeout=idle, first_chunk_timeout=first, heartbeat_seconds=hb,
                ):
                    if frame.get("error"):
                        saw_error_frame = True
                    yield frame
                if turn.errored:
                    completed_cleanly = True
                    return
                if not turn.had_cc_call:
                    # 无 cc 调用：终轮，补发 finish_reason / 非 cc tool_calls
                    final_delta: dict[str, Any] = {}
                    if turn.accumulated_tool_calls:
                        final_delta["tool_calls"] = turn.accumulated_tool_calls
                    yield {"choices": [{"index": 0, "delta": final_delta,
                                        "finish_reason": turn.finish_reason or "stop"}]}
                else:
                    async for frame in stream_cross_consult_continuation(
                        initial_tool_calls=turn.accumulated_tool_calls,
                        body=body, source_provider=provider, config=self.config,
                        cc_config=self.config.cross_consult, accumulator=accumulator,
                    ):
                        if frame.get("error"):
                            saw_error_frame = True
                        yield frame
                completed_cleanly = True
            else:
                async for chunk_dict in iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                ):
                    if isinstance(chunk_dict.get("error"), dict) and not chunk_dict.get("choices"):
                        saw_error_frame = True
                    yield chunk_dict
                completed_cleanly = True
        finally:
            accumulator.flush_to_cache(self._reasoning_cache)
            if completed_cleanly and not saw_error_frame:
                self._commit_pending_upgrade(body)
```

> 删除原 `buffered_chunks` / `accumulated_tool_calls` 局部变量声明（`520-521`）——现已由 `TurnResult` 承载。保留 `request_messages` / `accumulator` / `completed_cleanly` / `saw_error_frame` 声明。

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_cross_consult_loop.py::test_iter_chat_chunks_streams_cross_consult_live -v`
Expected: PASS

- [ ] **Step 6: Add the remaining integration assertions**

在 `tests/test_cross_consult_loop.py` 追加（复用上一个测试的 patch 骨架）：

```python
async def test_iter_chat_chunks_no_cc_call_passes_through(cfg_cross):
    """初始流不含 cc 调用：content 透传 + 终轮 finish_reason，行为等价直通。"""
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    provider = cfg_cross.providers["deepseek"]

    async def plain(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "你好"},
                            "finish_reason": "stop"}]}

    with patch("deep_proxy.router.iter_litellm_chunks", new=plain):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hi"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]
    assert any(fr.get("choices", [{}])[0].get("delta", {}).get("content") == "你好"
               for fr in frames)
    assert any(fr.get("choices", [{}])[0].get("finish_reason") == "stop" for fr in frames)


async def test_iter_chat_chunks_heartbeat_during_consult(cfg_cross):
    """consult 执行慢时，客户端收到心跳帧。"""
    import asyncio
    from unittest.mock import patch
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_cross)
    # 心跳间隔调小以便快速触发
    router.config.cross_consult.stream_heartbeat_seconds = 1
    provider = cfg_cross.providers["deepseek"]

    async def initial_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "cc1", "type": "function",
             "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}]},
            "finish_reason": "tool_calls"}]}

    async def resend_stream(config, body, *, _accumulator=None, provider=None):
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}

    calls = {"n": 0}
    def dispatch(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        return initial_stream(config, body) if calls["n"] == 1 else resend_stream(config, body)

    async def slow_consult(**kw):
        await asyncio.sleep(1.5)
        return "外部视角"

    with patch("deep_proxy.router.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", new=dispatch), \
         patch("deep_proxy.cross_consult.interceptor.execute_consult", new=slow_consult):
        body = {"model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "use cc"}]}
        body = await router.prepare_request(
            body, sampling_profile=cfg_cross.precise_sampling, provider=provider,
        )
        frames = [f async for f in router.iter_chat_chunks(body, provider=provider)]
    assert any(f == {"_dp_heartbeat": True} for f in frames)
```

- [ ] **Step 7: Run the cross_consult loop tests**

Run: `python -m pytest tests/test_cross_consult_loop.py -v`
Expected: PASS（含新增 3 项；旧的 intercept 单块测试已删）。

- [ ] **Step 8: Commit**

```bash
git add deep_proxy/router.py tests/test_cross_consult_loop.py
git commit -m "feat(cross-consult): iter_chat_chunks 改为客户端真流式透传 + continuation"
```

---

## Task 7: 清理死代码 `synthesize_final_stream_chunk`

接线改完后 `synthesize_final_stream_chunk` 不再被任何生产路径引用，删除函数 + 其专属单测。

**Files:**
- Modify: `deep_proxy/cross_consult/interceptor.py`（删除 `synthesize_final_stream_chunk`，`125-146`）
- Modify: `tests/test_cross_consult_loop.py`（删除 `test_streaming_final_chunk_includes_reasoning_content_when_present`）

- [ ] **Step 1: Verify no remaining references**

Run: `python -m pytest -q` 之前先确认引用已清。

Run: `git grep -n synthesize_final_stream_chunk`
Expected: 仅出现在 `interceptor.py` 定义处与 `test_cross_consult_loop.py` 的专属测试中（无 `router.py` 引用——Task 6 已移除 import 与调用）。若 `router.py` 仍有引用，回到 Task 6 修正。

- [ ] **Step 2: Delete the function**

删除 `deep_proxy/cross_consult/interceptor.py` 中整个 `synthesize_final_stream_chunk` 定义（含 docstring，约 `125-146`）。

- [ ] **Step 3: Delete its dedicated test**

删除 `tests/test_cross_consult_loop.py::test_streaming_final_chunk_includes_reasoning_content_when_present` 整个函数。

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS（全绿；无 `NameError`/`ImportError`）。

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/interceptor.py tests/test_cross_consult_loop.py
git commit -m "refactor(cross-consult): 删除被真流式取代的 synthesize_final_stream_chunk"
```

---

## Task 8: 全量回归 + 文档

**Files:**
- Modify: `CLAUDE.md`（cross_consult 段落补一句客户端真流式行为）

- [ ] **Step 1: Full suite green**

Run: `python -m pytest -q`
Expected: PASS（全部）。

- [ ] **Step 2: Update CLAUDE.md**

在 CLAUDE.md 的 Cross-Consult 段落（架构概览下方那段）末尾追加一句：

```
启用后流式 endpoint 对客户端真流式：content/reasoning 逐 token 透传、cross_consult 工具帧被抑制、consult 执行间隙发 SSE keep-alive 心跳（stream_heartbeat_seconds）。
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(cross-consult): 记录客户端真流式行为"
```

---

## Self-Review 结论（作者自检）

- **Spec 覆盖**：§3 三决策 → 心跳=Task 1/2/3 + with_heartbeat、前导透传=stream_one_turn content 透传（Task 4）、多 reasoning=stream_one_turn reasoning 透传 + continuation 跨轮（Task 4/5）。§4 三单元=Task 3/4/5。§4.4 接线=Task 6。§6 协议心跳=Task 2。§7 配置=Task 1。§8 缓存（复用 accumulator）=Task 5/6。§9 错误=stream_one_turn errored（Task 4）+ continuation/接线 return（Task 5/6）。§11 删 synthesize=Task 7。
- **占位符**：无 TBD；每个改码步骤含完整代码。
- **类型一致**：`TurnResult` 字段（accumulated_tool_calls/content/had_cc_call/finish_reason/errored）在 Task 4 定义、Task 5/6 使用一致；`_Done.value`、`stream_one_turn(chunk_iter, result, *, ...)` 签名跨 Task 一致；`stream_cross_consult_continuation(*, initial_tool_calls, body, source_provider, config, cc_config, accumulator)` 在 Task 5 定义、Task 6 调用一致。
- **风险点（实现时验证）**：§8 的 `process_response` parity——重发响应原走 dict 级 `process_response`，现走逐 delta `process_streaming_delta`；若发现 null 清理 / reasoning 字段差异，在 Task 6 补一个 per-frame 规整或在测试中补断言。
```
