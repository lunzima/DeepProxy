# cross_consult Retry + Hard-Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the cross_consult streaming path from the deprecated "请重试" notice to the proxy-side retry + hard-error model, sharing one generalized retry skeleton with the plain path.

**Architecture:** Extract the plain path's `stream_with_retry` into a generic `stream_turn_with_retry(make_attempt, …)` skeleton; `stream_with_retry` becomes a thin adapter. The cc initial turn and each resend turn wrap `stream_one_turn` in `make_attempt` closures (fresh upstream + accumulator snapshot/restore + budget clamp) and capture the winning `TurnResult` via an `on_result` callback. Delete the now-dead notice helpers and `STREAM_ERRORED`.

**Tech Stack:** Python 3.12, asyncio async generators, pytest (asyncio_mode=auto).

Spec: `docs/superpowers/specs/2026-06-04-cross-consult-retry-design.md`

---

### Task 1: Accumulator `snapshot()` / `restore()`

**Files:**
- Modify: `deep_proxy/compatibility/reasoning_handler.py` (class `StreamingReasoningAccumulator`, after `reset()` ~line 283)
- Test: `tests/test_reasoning_handler.py`

- [ ] **Step 1: Write the failing test**

```python
def test_accumulator_snapshot_restore_isolates_failed_attempt():
    from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator
    acc = StreamingReasoningAccumulator(request_messages=[{"role": "user", "content": "hi"}])
    acc.consume({"choices": [{"index": 0, "delta": {"content": "turn1"}}]})
    snap = acc.snapshot()
    # 失败尝试追加内容后回滚
    acc.consume({"choices": [{"index": 0, "delta": {"reasoning_content": "junk"}}]})
    acc.restore(snap)
    assert acc._slots[0]["content"] == "turn1"
    assert acc._slots[0]["reasoning_content"] == ""
    # snapshot 深拷贝：restore 后继续 consume 不污染已持有的 snap
    acc.consume({"choices": [{"index": 0, "delta": {"content": "2"}}]})
    assert snap[0]["content"] == "turn1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reasoning_handler.py::test_accumulator_snapshot_restore_isolates_failed_attempt -v`
Expected: FAIL with `AttributeError: 'StreamingReasoningAccumulator' object has no attribute 'snapshot'`

- [ ] **Step 3: Write minimal implementation**

In `reasoning_handler.py`, after the `reset()` method:

```python
    def snapshot(self) -> Dict[int, Dict[str, Any]]:
        """深拷贝当前 per-choice 槽，供流式重试在一轮开始时存档。"""
        return {i: {**s} for i, s in self._slots.items()}

    def restore(self, snap: Dict[int, Dict[str, Any]]) -> None:
        """回滚到 snapshot：丢弃 snapshot 之后（失败尝试）的累加，保留更早的轮次。
        深拷贝 snap 的槽，避免后续 consume 反向污染调用方持有的 snapshot。"""
        self._slots = {i: {**s} for i, s in snap.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reasoning_handler.py::test_accumulator_snapshot_restore_isolates_failed_attempt -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/compatibility/reasoning_handler.py tests/test_reasoning_handler.py
git commit -m "feat(reasoning): StreamingReasoningAccumulator snapshot/restore"
```

---

### Task 2: `stream_one_turn` gains `reasoning_idle`

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (`stream_one_turn` ~line 303-357)
- Test: `tests/test_cross_consult_client_stream.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_stream_one_turn_reasoning_idle_tolerates_gap():
    """显式 reasoning_idle：检测到 reasoning 后，超过 content idle 但 ≤ reasoning_idle
    的停顿不触发超时。"""
    async def reasoning_then_pause():
        yield _delta_chunk(reasoning_content="思考")
        await asyncio.sleep(0.35)          # > idle 0.15，< reasoning_idle 2.0
        yield _delta_chunk(content="答案")
        yield _finish_chunk("stop")

    res = TurnResult()
    out = [f async for f in stream_one_turn(
        reasoning_then_pause(), res, tool_name="cross_consult",
        idle_timeout=0.15, reasoning_idle=2.0,
        first_chunk_timeout=5.0, heartbeat_seconds=0.1,
    )]
    assert res.timed_out is False
    assert res.content == "答案"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_client_stream.py::test_stream_one_turn_reasoning_idle_tolerates_gap -v`
Expected: FAIL with `TypeError: stream_one_turn() got an unexpected keyword argument 'reasoning_idle'`

- [ ] **Step 3: Write minimal implementation**

In `stream_one_turn`, change the signature to add `reasoning_idle: float | None = None` (keyword), and replace the `_make_reasoning_aware_idle` call. Current head:

```python
async def stream_one_turn(
    chunk_iter: AsyncIterator[dict],
    result: TurnResult,
    *,
    tool_name: str,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
) -> AsyncGenerator[Any, None]:
```

becomes:

```python
async def stream_one_turn(
    chunk_iter: AsyncIterator[dict],
    result: TurnResult,
    *,
    tool_name: str,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    reasoning_idle: float | None = None,
) -> AsyncGenerator[Any, None]:
```

Replace the body line:

```python
    idle_ref, reasoning_idle = _make_reasoning_aware_idle(idle_timeout, first_chunk_timeout)
```

with:

```python
    reasoning_idle_val = (
        max(idle_timeout, reasoning_idle) if reasoning_idle is not None
        else compute_reasoning_idle(idle_timeout, first_chunk_timeout)
    )
    idle_ref = [idle_timeout]
```

and in the reasoning-bump block, rename `reasoning_idle` → `reasoning_idle_val`:

```python
            if _has_reasoning_content(chunk) and idle_ref[0] < reasoning_idle_val:
                idle_ref[0] = reasoning_idle_val
                logger.debug(
                    "stream_one_turn reasoning seen, idle %.0f→%.0f",
                    idle_timeout, reasoning_idle_val,
                )
```

- [ ] **Step 4: Run test + existing cc tests to verify pass**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -v`
Expected: PASS (new test + all existing; the `None` default preserves old behavior)

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_cross_consult_client_stream.py
git commit -m "feat(cc): stream_one_turn 接受显式 reasoning_idle"
```

---

### Task 3: Generalize `stream_turn_with_retry`; `stream_with_retry` → adapter

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (`stream_with_retry` ~line 448-515)
- Test: `tests/test_stream_retry.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_stream_turn_with_retry_on_result_captures_winning_turn():
    """泛化骨架：成功收尾时把 winning TurnResult 交给 on_result；committed 前的超时重试。"""
    from deep_proxy.cross_consult.client_stream import stream_turn_with_retry
    calls = {"n": 0}
    captured = {}

    def make_attempt(turn, remaining):
        calls["n"] += 1
        if calls["n"] == 1:
            async def stall():
                await asyncio.sleep(10.0)
                turn.timed_out = True  # 兜底（真实路径由 stream_one_turn 写）
                yield _delta_chunk(content="x")
            return stall()
        async def ok():
            turn.content = "hi"
            turn.finish_reason = "stop"
            yield _delta_chunk(content="hi")
            yield _finish_chunk("stop")
        return ok()

    out = [f async for f in stream_turn_with_retry(
        make_attempt, max_total_seconds=600.0,
        on_result=lambda t: captured.__setitem__("turn", t),
    )]
    # 第 1 次尝试需真正超时；改用确定性：见下方实现说明——这里用 spy 注入 timed_out。
    assert captured.get("turn") is not None
    assert captured["turn"].content == "hi"
    assert not any(is_error_frame(f) for f in out)
```

> Note: the first attempt must set `turn.timed_out=True` to trigger a retry. Since
> `make_attempt` controls the turn, set it inside the stalling branch BEFORE the
> sleep so the skeleton sees it after the (empty) stream drains. Replace the
> `stall()` body with: `turn.timed_out = True` then `return; yield` (an empty async
> gen), so the attempt yields nothing and the skeleton reads `turn.timed_out`.

Use this corrected `make_attempt` in the test:

```python
    def make_attempt(turn, remaining):
        calls["n"] += 1
        if calls["n"] == 1:
            async def stall():
                turn.timed_out = True
                return
                yield  # noqa
            return stall()
        async def ok():
            turn.content = "hi"
            turn.finish_reason = "stop"
            yield _delta_chunk(content="hi")
            yield _finish_chunk("stop")
        return ok()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stream_retry.py::test_stream_turn_with_retry_on_result_captures_winning_turn -v`
Expected: FAIL with `ImportError: cannot import name 'stream_turn_with_retry'`

- [ ] **Step 3: Write minimal implementation**

In `client_stream.py`, replace the entire body of `stream_with_retry` with the generic skeleton + adapter. Add the generic BEFORE `stream_with_retry`:

```python
async def stream_turn_with_retry(
    make_attempt: Callable[[TurnResult, float], AsyncIterator[dict]],
    *,
    max_total_seconds: float,
    on_result: Callable[[TurnResult], None] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[dict, None]:
    """通用 pre-content 重试 + 硬错误骨架（plain 与 cross_consult 共享）。

    make_attempt(turn, remaining) 产出**一次全新尝试**的帧流——调用方在其中接好
    turn-streamer（plain: stream_with_idle_timeout；cc: stream_one_turn）、全新上游、
    accumulator 重置/回滚、以及把 pre-content 预算钳到 remaining。

    非超时收尾（干净成功 / 真实 error frame 已透传）时调用 on_result(turn) 把**胜出轮**
    的 TurnResult 交还调用方，再 return；committed 后停顿 / 总预算耗尽 → 硬错误帧。
    见 docs/superpowers/specs/2026-06-04-cross-consult-retry-design.md。
    """
    deadline = now() + max_total_seconds
    committed = False
    while True:
        remaining = deadline - now()
        if remaining <= 0:
            yield make_hard_error_frame(
                f"上游持续无响应，超过 {max_total_seconds:g}s 总预算，本轮中断。"
            )
            return
        turn = TurnResult()
        async for frame in make_attempt(turn, remaining):
            if _frame_has_visible_output(frame):
                committed = True
            yield frame
        if not turn.timed_out:
            if on_result is not None:
                on_result(turn)
            return
        if committed:
            yield make_hard_error_frame(
                "已输出部分内容后上游中断，不可续传，本轮中断。"
            )
            return
        yield _HEARTBEAT
```

Then replace `stream_with_retry`'s body with the adapter (keep its docstring's first paragraph, drop the now-moved loop logic):

```python
async def stream_with_retry(
    make_upstream: Callable[[], AsyncIterator[dict]],
    *,
    idle_timeout: float,
    reasoning_idle: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    max_total_seconds: float,
    now: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[dict, None]:
    """plain（非 cross_consult）路径适配器：passthrough turn-streamer
    （stream_with_idle_timeout）+ pre-content 预算钳制，委托给 stream_turn_with_retry。"""
    def make_attempt(turn: TurnResult, remaining: float) -> AsyncIterator[dict]:
        return stream_with_idle_timeout(
            make_upstream(), result=turn,
            idle_timeout=idle_timeout,
            reasoning_idle=min(reasoning_idle, remaining),
            first_chunk_timeout=min(first_chunk_timeout, remaining),
            heartbeat_seconds=heartbeat_seconds,
        )
    async for frame in stream_turn_with_retry(
        make_attempt, max_total_seconds=max_total_seconds, now=now,
    ):
        yield frame
```

- [ ] **Step 4: Run test + full stream_retry suite**

Run: `python -m pytest tests/test_stream_retry.py -v`
Expected: PASS (new test + all 13 existing; the clamp test still patches `stream_with_idle_timeout`, reached via the adapter)

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_stream_retry.py
git commit -m "refactor(stream): 抽出通用 stream_turn_with_retry，stream_with_retry 降为适配器"
```

---

### Task 4: cc initial turn uses `stream_turn_with_retry`

**Files:**
- Modify: `deep_proxy/router.py` (`_iter_cc_chunks` ~line 605-651)
- Test: `tests/test_stream_retry.py`

- [ ] **Step 1: Write the failing tests**

```python
def _cc_body():
    return {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}


async def test_cc_initial_turn_retries_then_succeeds(router_dual, monkeypatch):
    """cc 初始轮 pre-content 挂死 → 代理重发 → 成功；无 notice / error frame。"""
    router = router_dual
    router.config.cross_consult.enabled = True
    router.config.cross_consult.pairs = {"deepseek": "mimo", "mimo": "deepseek"}
    router.config.cross_consult.first_chunk_timeout_seconds = 1
    router.config.cross_consult.call_timeout_seconds = 1
    router.config.cross_consult.stream_heartbeat_seconds = 1
    calls = {"n": 0}

    def fake_iter(config, body, *, _accumulator=None, provider=None):
        n = calls["n"]; calls["n"] += 1
        if n == 0:
            async def stall():
                await asyncio.sleep(10.0)
                yield _delta_chunk(content="x")
            return stall()
        async def ok():
            yield _delta_chunk(content="答案")
            yield _finish_chunk("stop")
        return ok()

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    prov = router.config.providers["deepseek"]
    out = [f async for f in router.iter_chat_chunks(_cc_body(), provider=prov)]
    assert calls["n"] == 2
    assert _content_of(out, "答案")
    assert not any(is_error_frame(f) for f in out)
    assert not any("[DeepProxy]" in (d.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
                   for d in out if "choices" in d)


async def test_cc_initial_turn_post_content_stall_hard_errors(router_dual, monkeypatch):
    """cc 初始轮已输出 content 后挂死 → 硬错误帧，不重发。"""
    router = router_dual
    router.config.cross_consult.enabled = True
    router.config.cross_consult.pairs = {"deepseek": "mimo", "mimo": "deepseek"}
    router.config.cross_consult.first_chunk_timeout_seconds = 5
    router.config.cross_consult.call_timeout_seconds = 1
    router.config.cross_consult.stream_heartbeat_seconds = 1
    router.config.streaming.reasoning_idle_timeout_seconds = 1
    calls = {"n": 0}

    def fake_iter(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        async def gen():
            yield _delta_chunk(content="部分")
            await asyncio.sleep(10.0)
            yield _delta_chunk(content="never")
        return gen()

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    prov = router.config.providers["deepseek"]
    out = [f async for f in router.iter_chat_chunks(_cc_body(), provider=prov)]
    assert calls["n"] == 1
    assert _content_of(out, "部分")
    assert any(is_error_frame(f) for f in out)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_stream_retry.py -k cc_initial_turn -v`
Expected: FAIL — current `_iter_cc_chunks` does not retry (calls["n"]==1 on first test) and emits notice, no error frame.

- [ ] **Step 3: Rewrite `_iter_cc_chunks`**

Replace the body of `_iter_cc_chunks` (from `turn = TurnResult()` through the end) with:

```python
        cc = self.config.cross_consult
        sc = self.config.streaming
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult, remaining: float):
            accumulator.restore(snap)   # 丢弃失败尝试的累加，保留更早内容（初始轮 snap 为空）
            return stream_one_turn(
                iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                ),
                turn, tool_name=cc.tool_name,
                idle_timeout=float(cc.call_timeout_seconds),
                reasoning_idle=min(float(sc.reasoning_idle_timeout_seconds), remaining),
                first_chunk_timeout=min(float(cc.first_chunk_timeout_seconds), remaining),
                heartbeat_seconds=float(cc.stream_heartbeat_seconds),
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_total_seconds=float(sc.max_stream_total_seconds),
            on_result=lambda t: captured.__setitem__("turn", t),
        ):
            yield frame
        turn = captured.get("turn")
        if turn is None or turn.errored:
            # 硬错误（已发 error frame）或真实上游 error frame（已逐帧透传）→ 终止。
            return
        if not turn.had_cc_call:
            # 无 cc 调用：终轮，补发 finish_reason / 非 cc tool_calls
            yield make_terminal_frame(turn.finish_reason, turn.accumulated_tool_calls)
            return
        # 进入 continuation
        async for frame in stream_cross_consult_continuation(
            initial_tool_calls=turn.accumulated_tool_calls,
            body=body, source_provider=provider, config=self.config,
            cc_config=cc, accumulator=accumulator, initial_content=turn.content,
        ):
            yield frame
```

Update the imports in `router.py`: add `stream_turn_with_retry` to the `from .cross_consult.client_stream import (...)` block.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_stream_retry.py -k cc_initial_turn -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/router.py tests/test_stream_retry.py
git commit -m "feat(cc): 初始轮接入 stream_turn_with_retry（重试+硬错误）"
```

---

### Task 5: cc resend turn uses `stream_turn_with_retry`

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (`stream_cross_consult_continuation` resend block ~line 592-618)
- Test: `tests/test_stream_retry.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_cc_resend_turn_retries_then_succeeds(router_dual, monkeypatch):
    """cc 重发轮 pre-content 挂死 → 重发 → 成功。初始轮发起 cc 调用，consult 直接返回，
    第一次重发挂死，第二次成功。"""
    router = router_dual
    cc = router.config.cross_consult
    cc.enabled = True
    cc.pairs = {"deepseek": "mimo", "mimo": "deepseek"}
    cc.first_chunk_timeout_seconds = 1
    cc.call_timeout_seconds = 1
    cc.stream_heartbeat_seconds = 1
    calls = {"n": 0}

    def _cc_tool_chunk():
        return {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}]},
            "finish_reason": None}]}

    def fake_iter(config, body, *, _accumulator=None, provider=None):
        n = calls["n"]; calls["n"] += 1
        if n == 0:                       # 初始轮：发起 cc 调用
            async def init():
                yield _cc_tool_chunk()
                yield _finish_chunk("tool_calls")
            return init()
        if n == 1:                       # 第一次重发：pre-content 挂死
            async def stall():
                await asyncio.sleep(10.0)
                yield _delta_chunk(content="x")
            return stall()
        async def ok():                  # 第二次重发：成功
            yield _delta_chunk(content="终答")
            yield _finish_chunk("stop")
        return ok()

    async def fake_consult(tc, *, call_count, target_provider, config, cc_config):
        return ("咨询结果", True)

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    monkeypatch.setattr("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", fake_iter)
    monkeypatch.setattr("deep_proxy.cross_consult.client_stream.resolve_consult_tool_call", fake_consult)
    prov = router.config.providers["deepseek"]
    out = [f async for f in router.iter_chat_chunks(_cc_body(), provider=prov)]
    assert calls["n"] == 3               # 初始 + 2 次重发
    assert _content_of(out, "终答")
    assert not any(is_error_frame(f) for f in out)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_stream_retry.py::test_cc_resend_turn_retries_then_succeeds -v`
Expected: FAIL — resend does not retry (calls["n"]==2, no "终答").

- [ ] **Step 3: Rewrite the resend block in `stream_cross_consult_continuation`**

Replace the resend block (from `resend_iter = iter_litellm_chunks(...)` through `yield STREAM_ERRORED / return`):

```python
        # 重发：流式，逐 chunk 透传 + pre-content 重试；复用同一 accumulator 写缓存。
        sc = config.streaming
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult, remaining: float):
            accumulator.restore(snap)   # 丢弃失败重发的累加，保留更早轮次内容
            return stream_one_turn(
                iter_litellm_chunks(
                    config, body, _accumulator=accumulator, provider=source_provider,
                ),
                turn, tool_name=cc_config.tool_name,
                idle_timeout=idle,
                reasoning_idle=min(float(sc.reasoning_idle_timeout_seconds), remaining),
                first_chunk_timeout=min(first, remaining),
                heartbeat_seconds=hb,
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_total_seconds=float(sc.max_stream_total_seconds),
            on_result=lambda t: captured.__setitem__("turn", t),
        ):
            yield frame
        turn = captured.get("turn")
        if turn is None or turn.errored:
            return
        turn_tool_calls = turn.accumulated_tool_calls
        turn_content = turn.content
        # 终轮（无 cc 调用）：把本轮 finish_reason / 非 cc tool_calls 作为终结帧透传
        if not turn.had_cc_call:
            yield make_terminal_frame(turn.finish_reason, turn_tool_calls)
            return
```

(Note: `idle = float(cc_config.call_timeout_seconds)`, `first = float(cc_config.first_chunk_timeout_seconds)`, `hb = float(cc_config.stream_heartbeat_seconds)` are already bound at the top of the function — keep them.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_stream_retry.py::test_cc_resend_turn_retries_then_succeeds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_stream_retry.py
git commit -m "feat(cc): 重发轮接入 stream_turn_with_retry（重试+硬错误）"
```

---

### Task 6: snapshot/restore preserves cross-turn content on retry

**Files:**
- Test only: `tests/test_stream_retry.py`

- [ ] **Step 1: Write the failing-then-passing test** (guards the snapshot/restore wiring)

```python
async def test_cc_resend_retry_preserves_prior_turn_content_in_cache(router_dual, monkeypatch):
    """重发轮重试时，初始轮已累加的内容不得被 restore 丢弃——缓存须含初始前导 + 终答。"""
    from deep_proxy.compatibility.reasoning_handler import StreamingReasoningAccumulator
    router = router_dual
    cc = router.config.cross_consult
    cc.enabled = True
    cc.pairs = {"deepseek": "mimo", "mimo": "deepseek"}
    cc.first_chunk_timeout_seconds = 1
    cc.call_timeout_seconds = 1
    cc.stream_heartbeat_seconds = 1
    calls = {"n": 0}

    def _cc_tool_chunk():
        return {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "cross_consult", "arguments": '{"question":"q"}'}}]},
            "finish_reason": None}]}

    def fake_iter(config, body, *, _accumulator=None, provider=None):
        n = calls["n"]; calls["n"] += 1
        if n == 0:
            async def init():
                _accumulator.consume(_delta_chunk(content="前导"))
                yield _delta_chunk(content="前导")
                yield _cc_tool_chunk()
                yield _finish_chunk("tool_calls")
            return init()
        if n == 1:
            async def stall():
                await asyncio.sleep(10.0)
                yield _delta_chunk(content="x")
            return stall()
        async def ok():
            _accumulator.consume(_delta_chunk(content="终答"))
            yield _delta_chunk(content="终答")
            yield _finish_chunk("stop")
        return ok()

    async def fake_consult(tc, *, call_count, target_provider, config, cc_config):
        return ("结果", True)

    captured = {}
    orig_flush = StreamingReasoningAccumulator.flush_to_cache
    def spy_flush(self, cache):
        captured["content"] = self._slots.get(0, {}).get("content")
        return orig_flush(self, cache)

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    monkeypatch.setattr("deep_proxy.cross_consult.client_stream.iter_litellm_chunks", fake_iter)
    monkeypatch.setattr("deep_proxy.cross_consult.client_stream.resolve_consult_tool_call", fake_consult)
    monkeypatch.setattr(StreamingReasoningAccumulator, "flush_to_cache", spy_flush)
    prov = router.config.providers["deepseek"]
    out = [f async for f in router.iter_chat_chunks(_cc_body(), provider=prov)]
    assert captured["content"] == "前导终答"   # 初始前导保留 + 终答；失败重发的 'x' 不在
```

- [ ] **Step 2: Run to verify it passes** (Tasks 4-5 already wired snapshot/restore)

Run: `python -m pytest tests/test_stream_retry.py::test_cc_resend_retry_preserves_prior_turn_content_in_cache -v`
Expected: PASS (initial-turn snap is taken AFTER 前导 accumulates → restore in resend keeps 前导; failed resend's nothing-committed is discarded)

> If this FAILS with `content == "终答"` (前导 lost), the initial-turn `snapshot()`
> was taken too early or the resend `restore` over-clears — verify the resend's
> `snap = accumulator.snapshot()` runs AFTER the initial turn's content is in slots
> (it does: continuation runs after `_iter_cc_chunks` forwards the initial turn).

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_retry.py
git commit -m "test(cc): 重发重试保留初始轮内容入缓存"
```

---

### Task 7: Cleanup — delete notice helpers, remove dead `STREAM_ERRORED`, revert comments

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (delete `make_timeout_notice_frames`, `_timeout_notice_text`)
- Modify: `deep_proxy/router.py` (remove `STREAM_ERRORED` handling + import if dead; revert deprecation comment)
- Modify: `tests/test_cross_consult_client_stream.py` (delete obsolete helper tests)
- Modify: `tests/test_stream_timeout_notice.py` (rewrite cc test, update docstring)

- [ ] **Step 1: Verify the helpers are now dead**

Run:
```bash
grep -rn "make_timeout_notice_frames\|_timeout_notice_text" deep_proxy/
grep -rn "STREAM_ERRORED" deep_proxy/
```
Expected: only definitions + (for STREAM_ERRORED) any remaining producer/consumer. After Tasks 4-5, `_iter_plain_chunks` and both cc sites no longer reference them. If `grep` shows production references remain, STOP and resolve before deleting.

- [ ] **Step 2: Delete `make_timeout_notice_frames` and `_timeout_notice_text`**

Remove both functions from `client_stream.py` (lines ~156-185). Update the module docstring's sentinel list (lines ~11-14) if it mentions notice frames.

- [ ] **Step 3: Remove `STREAM_ERRORED` if fully dead**

If Step 1 showed no remaining producer of `STREAM_ERRORED` (both cc sites migrated, `_iter_plain_chunks` already dropped it), remove:
- the `STREAM_ERRORED` definition in `client_stream.py` (~line 42),
- the handler branch in `router.py iter_chat_chunks` (~line 590-594: `if frame is STREAM_ERRORED: saw_error_frame = True; continue`),
- `STREAM_ERRORED` from `router.py`'s import block and the `client_stream.py` `__all__`/docstring if present.

If a producer remains, keep it and add a one-line comment naming the remaining producer.

- [ ] **Step 4: Revert the deprecation comments from commit 559c945**

In `client_stream.py` `TurnResult` (the timeout-metadata comment) and `router.py` `_iter_cc_chunks`: these blocks are being deleted/rewritten by Tasks 4 and 7 anyway. Confirm no "已废弃，仅 cc 路径仍在用" comment remains referencing a now-deleted helper.

- [ ] **Step 5: Delete / rewrite obsolete tests**

In `tests/test_cross_consult_client_stream.py`: delete `test_make_timeout_notice_frames_first_chunk` and `test_make_timeout_notice_frames_mid_stream_distinct_text`, and remove `make_timeout_notice_frames` from the import on line ~37.

In `tests/test_stream_timeout_notice.py`: rewrite `test_cc_initial_turn_timeout_emits_graceful_notice` →

```python
async def test_cc_initial_turn_timeout_hard_errors():
    """cross_consult 激活：初始轮首 chunk 超时（预算耗尽）→ 硬错误帧（透传给客户端），
    不再注入已废弃的优雅通知。"""
    router, provider = _cc_router()
    router.config.streaming.max_stream_total_seconds = 1

    async def hang_iter(config, body, *, _accumulator=None, provider=None):
        await asyncio.sleep(5.0)
        yield {"choices": [{"index": 0, "delta": {"content": "never"}, "finish_reason": None}]}

    body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
    committed = {"hit": False}
    with patch("deep_proxy.router.iter_litellm_chunks", new=hang_iter), \
         patch.object(router, "_commit_pending_upgrade",
                      side_effect=lambda *a, **k: committed.__setitem__("hit", True)):
        out = [f async for f in router.iter_chat_chunks(body, provider=provider)]

    assert not _notice_present(out)
    assert not _clean_finish_present(out)
    assert not _no_error_frame(out)        # 硬错误帧存在
    assert committed["hit"] is False
```

Update the module docstring of `test_stream_timeout_notice.py` to drop the "cc 路径仍走旧通知" sentence.

- [ ] **Step 6: Run the full affected suite**

Run:
```bash
python -m pytest tests/test_stream_retry.py tests/test_cross_consult_client_stream.py tests/test_stream_timeout_notice.py tests/test_reasoning_handler.py tests/test_cross_consult_streaming.py -v
```
Expected: PASS (no references to deleted symbols).

- [ ] **Step 7: Full suite + commit**

Run: `python -m pytest -q`
Expected: all pass.

```bash
git add -A
git commit -m "refactor(cc): 删除废弃 notice 助手 + STREAM_ERRORED 死代码，重写 cc 超时测试"
```

---

## Self-Review notes

- **Spec coverage:** Task 1 = snapshot/restore (§4); Task 2 = stream_one_turn reasoning_idle (§2); Task 3 = generic skeleton + adapter (§1); Task 4 = cc initial turn (§3); Task 5 = cc resend turn (§3); Task 6 = cross-turn cache preservation (§4 test); Task 7 = cleanup (§5) + test updates (§Testing). All spec sections covered.
- **Type consistency:** `stream_turn_with_retry(make_attempt, *, max_total_seconds, on_result, now)`, `make_attempt(turn, remaining)`, `on_result(turn)`, `snapshot()→dict`, `restore(snap)` used identically across Tasks 1, 3, 4, 5.
- **Gated deletion:** Task 7 Step 1/3 gate the `STREAM_ERRORED` removal on a grep showing no remaining producer.
