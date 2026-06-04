# Timeout Mechanism Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the timeout system to a single rule — "any upstream call: `idle`s of no chunk (first chunk uses `first_chunk`) = stall; pre-content stalls re-issue up to `max_retries`, else/post-content = hard error; a stream still producing output is never interrupted" — by (step 1) replacing the per-attempt budget clamp + `max_stream_total_seconds` with a retry **count**, and (step 2) unifying the two timeout engines into one and merging the two config sets into one.

**Architecture:** `consume_with_heartbeat` becomes the single timeout/reasoning engine (owns first-chunk budget, content idle, and the reasoning-aware idle raise). Its three consumers — `stream_with_idle_timeout` (plain passthrough), `stream_one_turn` (cc accumulate+suppress), `aggregate_stream_to_response` (internal dict aggregation) — keep only their per-chunk handling. `stream_turn_with_retry` becomes count-based (no deadline/clamp). cross_consult timeouts are read from the single `StreamingConfig`.

**Tech Stack:** Python 3.12, asyncio async generators, pytest (asyncio_mode=auto).

Source analysis (the "current mechanism" this plan replaces) is in the conversation; key fixes it delivers: healthy reasoning streams are never walled near a deadline; cc's previously-inert `reasoning_idle`; three divergent reasoning-idle values (45/60/120) collapse to one; two engines → one; eight timeout knobs → ~four.

## File structure / responsibilities after this plan

- `deep_proxy/config.py` — `StreamingConfig`: single source of timeout truth (`first_chunk_timeout_seconds`, `idle_timeout_seconds`, `reasoning_idle_timeout_seconds`, `max_retries`, `heartbeat_seconds`). `max_stream_total_seconds` removed.
- `deep_proxy/cross_consult/config.py` — `CrossConsultConfig`: keeps only cross_consult semantics (`pairs`, `tool_name`, `max_calls_per_request`, prompts, redirect). Timeout fields removed.
- `deep_proxy/cross_consult/client_stream.py` — `consume_with_heartbeat` (THE engine: first-chunk + idle + reasoning raise); thin consumers `stream_with_idle_timeout`, `stream_one_turn`; `stream_turn_with_retry` (count-based) + `stream_with_retry` adapter.
- `deep_proxy/cross_consult/streaming.py` — `aggregate_stream_to_response` drives the engine (no second timeout loop).
- `deep_proxy/cross_consult/reasoning_idle.py` — `chunk_has_reasoning` stays; `compute_reasoning_idle` removed if it becomes unused.

---

### Task 1: StreamingConfig — `max_stream_total_seconds` → `max_retries`

**Files:**
- Modify: `deep_proxy/config.py` (`StreamingConfig` ~line 525-560)
- Test: `tests/test_stream_retry.py`

- [ ] **Step 1: Update the config defaults test**

Replace the body of `test_streaming_config_new_defaults` in `tests/test_stream_retry.py`:

```python
def test_streaming_config_new_defaults():
    sc = StreamingConfig()
    assert sc.idle_timeout_seconds == 15
    assert sc.reasoning_idle_timeout_seconds == 45
    assert sc.first_chunk_timeout_seconds == 120
    assert sc.max_retries == 2                       # 替代 max_stream_total_seconds
    assert sc.heartbeat_seconds == 10
    assert not hasattr(sc, "max_stream_total_seconds")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stream_retry.py::test_streaming_config_new_defaults -v`
Expected: FAIL (`max_retries` missing / `max_stream_total_seconds` still present)

- [ ] **Step 3: Edit `StreamingConfig`**

In `deep_proxy/config.py`, remove the `max_stream_total_seconds` Field and add:

```python
    max_retries: int = Field(
        default=2, ge=0, le=20,
        description="pre-content stall（首 chunk 前 / 推理中、尚无可见 content/tool_calls）"
                    "时重发原请求的最大次数。健康流（持续产出）不触发重试，故此计数只约束"
                    "dead-air 重发；耗尽 → 发硬错误帧。post-content stall 不重试。",
    )
```

Update the `StreamingConfig` class docstring: replace the `max_stream_total_seconds` bullet with a `max_retries` bullet describing count-based retry and that healthy streams are never interrupted.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stream_retry.py::test_streaming_config_new_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/config.py tests/test_stream_retry.py
git commit -m "refactor(config): StreamingConfig max_stream_total_seconds → max_retries"
```

---

### Task 2: `consume_with_heartbeat` owns the reasoning-aware idle raise

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (`_resolve_idle` ~160, `consume_with_heartbeat` ~167-252)
- Test: `tests/test_cross_consult_client_stream.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cross_consult_client_stream.py`:

```python
from deep_proxy.cross_consult.client_stream import consume_with_heartbeat, _Timeout, _HEARTBEAT


async def test_consume_with_heartbeat_raises_idle_on_reasoning():
    """引擎内部自适应：见 reasoning_content 后 idle 升到 reasoning_idle；
    推理停顿 > content idle 但 ≤ reasoning_idle 不超时。"""
    async def gen():
        yield _delta_chunk(reasoning_content="想")
        await asyncio.sleep(0.3)          # > idle 0.1，< reasoning_idle 2.0
        yield _delta_chunk(content="答")

    items = [it async for it in consume_with_heartbeat(
        gen(), idle_timeout=0.1, reasoning_idle=2.0,
        first_chunk_timeout=5.0, heartbeat_seconds=0.05, log_label="t",
    )]
    assert not any(isinstance(it, _Timeout) for it in items)
    chunks = [it for it in items if isinstance(it, dict) and "choices" in it]
    assert any(c["choices"][0]["delta"].get("content") == "答" for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_consult_client_stream.py::test_consume_with_heartbeat_raises_idle_on_reasoning -v`
Expected: FAIL with `TypeError: consume_with_heartbeat() got an unexpected keyword argument 'reasoning_idle'`

- [ ] **Step 3: Rewrite `consume_with_heartbeat` to own the raise; delete `_resolve_idle`**

Delete the `_resolve_idle` function. Replace `consume_with_heartbeat`'s signature and body:

```python
async def consume_with_heartbeat(
    chunk_iter: AsyncIterator[dict],
    *,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    log_label: str,
    reasoning_idle: float | None = None,
) -> AsyncGenerator[Any, None]:
    """单一超时引擎：消费上游 chunk 流，产出 chunk dict / `_HEARTBEAT` / `_Timeout(phase,seconds)`。

    自持有全部超时与 reasoning 自适应逻辑：
      - 首 chunk 前用 first_chunk_timeout；其后用 idle（content 阶段）。
      - 首次见到非空 reasoning_content 后，把 idle 升到 max(idle, reasoning_idle)
        （深度思考 burst 间隙属正常）。reasoning_idle=None → 不升级（content idle 全程）。
      - budget<=0 表示禁用该阶段超时（永不 trip）。
    StopAsyncIteration → 静默 return。finally cancel/drain/aclose 上游（确定性释放连接）。
    调用方契约：在自身 finally 里 `await gen.aclose()` 本生成器。
    """
    it = chunk_iter.__aiter__() if hasattr(chunk_iter, "__aiter__") else chunk_iter
    got_first = False
    current_idle = idle_timeout
    task: asyncio.Future = asyncio.ensure_future(it.__anext__())
    waited = 0.0
    try:
        while True:
            budget = current_idle if got_first else first_chunk_timeout
            step = heartbeat_seconds
            if budget and budget > 0:
                step = min(heartbeat_seconds, max(0.0, budget - waited))
            done, _ = await asyncio.wait(
                {task}, timeout=step if step > 0 else heartbeat_seconds,
            )
            if task not in done:
                waited += step
                if budget and budget > 0 and waited >= budget:
                    phase = "first_chunk" if not got_first else "mid_stream"
                    logger.warning("%s %s timeout after %.1fs", log_label, phase, budget)
                    yield _Timeout(phase, budget)
                    return
                yield _HEARTBEAT
                continue
            try:
                chunk = task.result()
            except StopAsyncIteration:
                return
            got_first = True
            waited = 0.0
            # reasoning 自适应：首见深度思考 token 后升级 idle 预算
            if (reasoning_idle is not None and current_idle < reasoning_idle
                    and _has_reasoning_content(chunk)):
                current_idle = max(current_idle, reasoning_idle)
                logger.debug("%s reasoning seen, idle→%.0f", log_label, current_idle)
            task = asyncio.ensure_future(it.__anext__())
            yield chunk
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except BaseException:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cross_consult_client_stream.py::test_consume_with_heartbeat_raises_idle_on_reasoning -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py tests/test_cross_consult_client_stream.py
git commit -m "refactor(engine): consume_with_heartbeat 自持 reasoning-aware idle，删 _resolve_idle"
```

---

### Task 3: `stream_with_idle_timeout` + `stream_one_turn` become thin (no own bump)

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (`stream_with_idle_timeout`, `stream_one_turn`)
- Test: existing tests in `tests/test_cross_consult_client_stream.py` + `tests/test_stream_retry.py` must stay green.

- [ ] **Step 1: Run the existing reasoning tests to confirm current green baseline**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -k "reasoning_idle or detection_only or forwards" -v`
Expected: PASS (baseline before refactor)

- [ ] **Step 2: Simplify `stream_with_idle_timeout`**

Replace the `reasoning_idle_val`/`idle_ref` computation and the in-loop reasoning bump. New body of the relevant region:

```python
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_with_idle_timeout",
        reasoning_idle=reasoning_idle,
    )
    saw_finish = False
    try:
        async for item in gen:
            if isinstance(item, _Timeout):
                if saw_finish:
                    return
                result.errored = True
                result.timed_out = True
                result.timeout_phase = item.phase
                result.timeout_seconds = item.seconds
                return
            if item is _HEARTBEAT:
                yield item
                continue
            chunk = item
            if any(c.get("finish_reason") for c in (chunk.get("choices") or [])):
                saw_finish = True
            yield chunk
    finally:
        await gen.aclose()
```

(The function keeps its `reasoning_idle: float | None = None` param; it now just forwards it. Delete the local `reasoning_idle_val`/`idle_ref` lines and the `_has_reasoning_content` bump block.)

- [ ] **Step 3: Simplify `stream_one_turn`**

Same change: delete its `reasoning_idle_val`/`idle_ref` computation and in-loop bump; pass `reasoning_idle=reasoning_idle` into `consume_with_heartbeat`. The per-chunk handling (`_accumulate_turn`, `_client_facing_chunk`, error-frame branch) is unchanged:

```python
    gen = consume_with_heartbeat(
        chunk_iter, idle_timeout=idle_timeout, first_chunk_timeout=first_chunk_timeout,
        heartbeat_seconds=heartbeat_seconds, log_label="stream_one_turn",
        reasoning_idle=reasoning_idle,
    )
    try:
        async for item in gen:
            if isinstance(item, _Timeout):
                result.errored = True
                result.timed_out = True
                result.timeout_phase = item.phase
                result.timeout_seconds = item.seconds
                return
            if item is _HEARTBEAT:
                yield item
                continue
            chunk = item
            if is_error_frame(chunk):
                result.errored = True
                yield chunk
                return
            _accumulate_turn(chunk, result, tool_name)
            fwd = _client_facing_chunk(chunk)
            if fwd is not None:
                yield fwd
    finally:
        await gen.aclose()
```

- [ ] **Step 4: Run the relevant suites**

Run: `python -m pytest tests/test_cross_consult_client_stream.py -v`
Expected: PASS (reasoning_idle tolerance + detection-only + passthrough all still hold via the engine)

- [ ] **Step 5: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py
git commit -m "refactor(engine): stream_with_idle_timeout/stream_one_turn 降为薄消费者（bump 收归引擎）"
```

---

### Task 4: `stream_turn_with_retry` → count-based; drop clamp/deadline

**Files:**
- Modify: `deep_proxy/cross_consult/client_stream.py` (`stream_turn_with_retry`, `stream_with_retry`)
- Modify: `deep_proxy/router.py` (`_iter_plain_chunks`)
- Test: `tests/test_stream_retry.py`

- [ ] **Step 1: Replace clamp/deadline tests with count + healthy-survives tests**

In `tests/test_stream_retry.py`: DELETE `test_hard_error_on_budget_exhaustion` and `test_retry_clamps_pre_content_budget_to_remaining` (they assert the removed clamp/deadline). Add:

```python
async def test_stream_turn_with_retry_count_based_exhaustion():
    """pre-content 连续 stall，重发达 max_retries 后发硬错误帧（无 deadline/now）。"""
    from deep_proxy.cross_consult.client_stream import stream_turn_with_retry, TurnResult
    calls = {"n": 0}

    def make_attempt(turn, _unused=None):
        calls["n"] += 1
        async def stall():
            turn.timed_out = True
            return
            yield  # noqa
        return stall()

    out = [f async for f in stream_turn_with_retry(make_attempt, max_retries=2)]
    assert calls["n"] == 3                    # 1 初次 + 2 重发
    assert any(is_error_frame(f) for f in out)


async def test_stream_turn_with_retry_healthy_stream_never_walled():
    """健康流（产出 content 后干净 finish）一次成功，不重试、无硬错误。"""
    from deep_proxy.cross_consult.client_stream import stream_turn_with_retry, TurnResult
    calls = {"n": 0}

    def make_attempt(turn, _unused=None):
        calls["n"] += 1
        async def ok():
            yield _delta_chunk(content="hi")
            yield _finish_chunk("stop")
        return ok()

    captured = {}
    out = [f async for f in stream_turn_with_retry(
        make_attempt, max_retries=2, on_result=lambda t: captured.__setitem__("t", t))]
    assert calls["n"] == 1
    assert captured.get("t") is not None
    assert not any(is_error_frame(f) for f in out)
```

Keep `test_stream_turn_with_retry_on_result_captures_winning_turn`, `test_retry_succeeds_after_pre_content_stall`, `test_hard_error_on_post_content_stall`, `test_real_upstream_error_forwarded_no_retry`, `test_clean_finish_succeeds_first_try` but update their `stream_with_retry(...)` / `stream_turn_with_retry(...)` calls to drop `max_total_seconds=`/`now=` and pass `max_retries=2` (and make_attempt signature `(turn, _)` — the second positional is no longer `remaining`; see Step 3).

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_stream_retry.py -k "count_based or never_walled" -v`
Expected: FAIL (`stream_turn_with_retry` still requires `max_total_seconds`)

- [ ] **Step 3: Rewrite `stream_turn_with_retry` and `stream_with_retry`**

```python
async def stream_turn_with_retry(
    make_attempt: Callable[[TurnResult], AsyncIterator[dict]],
    *,
    max_retries: int,
    on_result: Callable[[TurnResult], None] | None = None,
) -> AsyncGenerator[dict, None]:
    """通用 pre-content 重试 + 硬错误骨架（plain 与 cross_consult 共享）。

    make_attempt(turn) 产出**一次全新尝试**的帧流（每尝试自然 idle/first_chunk 窗口，
    不钳制——健康流永不被打断，只有 dead-air 才 stall）。据收尾决策：
      - 非超时收尾（干净成功 / 真实 error frame 已透传）：on_result(turn)，return。
      - 超时且已提交可见输出（committed）：post-content 不可续传 → 硬错误帧、return。
      - 超时且 pre-content 且重发次数已达 max_retries：硬错误帧、return。
      - 超时且 pre-content 且还可重发：心跳后重发（retries 计数 +1）。
    committed 一经置位不复位。见 docs/superpowers/plans/2026-06-04-timeout-simplification.md。
    """
    committed = False
    retries = 0
    while True:
        turn = TurnResult()
        async for frame in make_attempt(turn):
            if _frame_has_visible_output(frame):
                committed = True
            yield frame
        if not turn.timed_out:
            if on_result is not None:
                on_result(turn)
            return
        if committed:
            yield make_hard_error_frame("已输出部分内容后上游中断，不可续传，本轮中断。")
            return
        if retries >= max_retries:
            yield make_hard_error_frame(
                f"上游持续无响应，已重发 {retries} 次仍未恢复，本轮中断。")
            return
        retries += 1
        yield _HEARTBEAT


async def stream_with_retry(
    make_upstream: Callable[[], AsyncIterator[dict]],
    *,
    idle_timeout: float,
    reasoning_idle: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
    max_retries: int,
) -> AsyncGenerator[dict, None]:
    """plain 路径适配器：passthrough turn-streamer + 自然超时窗口（不钳制）。"""
    def make_attempt(turn: TurnResult) -> AsyncIterator[dict]:
        return stream_with_idle_timeout(
            make_upstream(), result=turn,
            idle_timeout=idle_timeout, reasoning_idle=reasoning_idle,
            first_chunk_timeout=first_chunk_timeout, heartbeat_seconds=heartbeat_seconds,
        )
    async for frame in stream_turn_with_retry(make_attempt, max_retries=max_retries):
        yield frame
```

Remove the now-unused `import time` if nothing else in the module uses it (grep first).

- [ ] **Step 4: Update `_iter_plain_chunks` in `router.py`**

Replace the `stream_with_retry(...)` call args:

```python
        async for chunk_dict in stream_with_retry(
            make_upstream,
            idle_timeout=float(sc.idle_timeout_seconds),
            reasoning_idle=float(sc.reasoning_idle_timeout_seconds),
            first_chunk_timeout=float(sc.first_chunk_timeout_seconds),
            heartbeat_seconds=float(sc.heartbeat_seconds),
            max_retries=int(sc.max_retries),
        ):
            yield chunk_dict
```

- [ ] **Step 5: Run the full stream_retry + plain timeout suites**

Run: `python -m pytest tests/test_stream_retry.py tests/test_stream_timeout_notice.py -v`
Expected: PASS. Note: `test_plain_path_first_chunk_timeout_retries_then_hard_errors` in `test_stream_timeout_notice.py` sets `max_stream_total_seconds=1` — change it to `router.config.streaming.max_retries = 1` and keep `first_chunk_timeout_seconds = 1` so it stalls fast and exhausts after 1 retry.

- [ ] **Step 6: Commit**

```bash
git add deep_proxy/cross_consult/client_stream.py deep_proxy/router.py tests/test_stream_retry.py tests/test_stream_timeout_notice.py
git commit -m "refactor(retry): stream_turn_with_retry 改 count-based(max_retries)，去 clamp/deadline；健康流不再被墙"
```

---

### Task 5: cc initial + resend turns use `max_retries`, natural windows

**Files:**
- Modify: `deep_proxy/router.py` (`_iter_cc_chunks` make_attempt)
- Modify: `deep_proxy/cross_consult/client_stream.py` (`stream_cross_consult_continuation` resend make_attempt)
- Test: `tests/test_stream_retry.py` cc tests

- [ ] **Step 1: Update cc retry tests for max_retries / natural windows**

In `tests/test_stream_retry.py`, the cc tests (`test_cc_initial_turn_*`, `test_cc_resend_turn_retries_then_succeeds`, `test_cc_resend_retry_preserves_prior_turn_content_in_cache`) currently rely on `max_stream_total_seconds`/clamp indirectly. They only set cc timeouts + assert retry counts, so they keep working IF the call sites pass `max_retries`. For the budget-exhaustion-style assertion, add:

```python
async def test_cc_initial_turn_count_based_hard_error(router_dual, monkeypatch):
    """cc 初始轮连续 pre-content stall，达 max_retries 后硬错误。"""
    router = router_dual
    cc = router.config.cross_consult
    cc.enabled = True
    cc.pairs = {"deepseek": "mimo", "mimo": "deepseek"}
    router.config.streaming.first_chunk_timeout_seconds = 1
    router.config.streaming.max_retries = 1
    router.config.streaming.heartbeat_seconds = 1
    calls = {"n": 0}

    def fake_iter(config, body, *, _accumulator=None, provider=None):
        calls["n"] += 1
        async def stall():
            await asyncio.sleep(10.0)
            yield _delta_chunk(content="x")
        return stall()

    monkeypatch.setattr("deep_proxy.router.iter_litellm_chunks", fake_iter)
    prov = router.config.providers["deepseek"]
    out = [f async for f in router.iter_chat_chunks(_cc_body(), provider=prov)]
    assert calls["n"] == 2                     # 1 初次 + 1 重发
    assert any(is_error_frame(f) for f in out)
```

(Note: after Task 6 the cc timeouts come from `StreamingConfig`, so this test sets `streaming.first_chunk_timeout_seconds`/`max_retries`. Until Task 6, the cc path still reads `cc.first_chunk_timeout_seconds`; set BOTH in this test so it passes before and after Task 6: also set `cc.first_chunk_timeout_seconds = 1`.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_stream_retry.py::test_cc_initial_turn_count_based_hard_error -v`
Expected: FAIL (cc make_attempt still uses `max_total_seconds`/clamp)

- [ ] **Step 3: Update `_iter_cc_chunks` make_attempt (router.py)**

```python
        cc = self.config.cross_consult
        sc = self.config.streaming
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult):
            accumulator.restore(snap)
            return stream_one_turn(
                iter_litellm_chunks(
                    self.config, body, _accumulator=accumulator, provider=provider,
                ),
                turn, tool_name=cc.tool_name,
                idle_timeout=float(sc.idle_timeout_seconds),
                reasoning_idle=float(sc.reasoning_idle_timeout_seconds),
                first_chunk_timeout=float(sc.first_chunk_timeout_seconds),
                heartbeat_seconds=float(sc.heartbeat_seconds),
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_retries=int(sc.max_retries),
            on_result=lambda t: captured.__setitem__("turn", t),
        ):
            yield frame
```

(The post-loop `turn = captured.get("turn"); if turn is None or turn.errored: return` etc. is unchanged.)

- [ ] **Step 4: Update the resend make_attempt (`stream_cross_consult_continuation`, client_stream.py)**

```python
        sc = config.streaming
        snap = accumulator.snapshot()
        captured: dict[str, TurnResult] = {}

        def make_attempt(turn: TurnResult):
            accumulator.restore(snap)
            return stream_one_turn(
                iter_litellm_chunks(
                    config, body, _accumulator=accumulator, provider=source_provider,
                ),
                turn, tool_name=cc_config.tool_name,
                idle_timeout=float(sc.idle_timeout_seconds),
                reasoning_idle=float(sc.reasoning_idle_timeout_seconds),
                first_chunk_timeout=float(sc.first_chunk_timeout_seconds),
                heartbeat_seconds=float(sc.heartbeat_seconds),
            )

        async for frame in stream_turn_with_retry(
            make_attempt, max_retries=int(sc.max_retries),
            on_result=lambda t: captured.__setitem__("turn", t),
        ):
            yield frame
```

The `idle = float(cc_config.call_timeout_seconds)` / `first = ...` / `hb = ...` bindings at the top of `stream_cross_consult_continuation` are now unused by the resend (still used by `with_heartbeat` for the consult call — see Task 6 Step 4). Leave them until Task 6 repoints them.

- [ ] **Step 5: Run cc retry suite**

Run: `python -m pytest tests/test_stream_retry.py -k cc -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_proxy/router.py deep_proxy/cross_consult/client_stream.py tests/test_stream_retry.py
git commit -m "refactor(cc): 初始/重发轮改 max_retries + 自然窗口（不钳制）"
```

---

### Task 6: Merge config — cc reads `StreamingConfig`; drop cc timeout fields; unify `aggregate_stream_to_response` onto the engine

**Files:**
- Modify: `deep_proxy/cross_consult/config.py` (remove 3 timeout fields)
- Modify: `deep_proxy/cross_consult/streaming.py` (drive `consume_with_heartbeat`)
- Modify: `deep_proxy/cross_consult/executor.py`, `deep_proxy/router.py` (cc non-stream paths) — pass `StreamingConfig` timeouts
- Modify: `config.example.yaml`
- Test: `tests/test_cross_consult_streaming.py`, `tests/test_cross_consult_config.py`

- [ ] **Step 1: Write a failing test for the unified aggregate engine**

In `tests/test_cross_consult_streaming.py`, add (and confirm the existing reasoning-idle tests still pass after the rewrite):

```python
async def test_aggregate_drives_engine_reasoning_then_content():
    """aggregate 走统一引擎：reasoning 后 idle 升级，停顿 ≤ reasoning_idle 不超时，聚合成 dict。"""
    from deep_proxy.cross_consult.streaming import aggregate_stream_to_response
    from deep_proxy.config import ProxyConfig, normalize_legacy_config

    cfg = ProxyConfig.model_validate(normalize_legacy_config({
        "deepseek": {"api_key": "sk", "api_base": "https://api.deepseek.com"}}))

    async def gen(config, body, *, provider=None):
        yield {"choices": [{"index": 0, "delta": {"reasoning_content": "想"}}]}
        await asyncio.sleep(0.3)        # > idle 0.1, < reasoning_idle 2.0
        yield {"choices": [{"index": 0, "delta": {"content": "答"}}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    resp = await aggregate_stream_to_response(
        cfg, {"model": "m", "messages": []}, provider=None,
        idle_timeout=0.1, reasoning_idle=2.0, first_chunk_timeout=5.0,
        heartbeat_seconds=0.05, iter_fn=gen,
    )
    assert "_dp_error" not in resp
    assert resp["choices"][0]["message"]["content"] == "答"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_cross_consult_streaming.py::test_aggregate_drives_engine_reasoning_then_content -v`
Expected: FAIL (`aggregate_stream_to_response` has no `reasoning_idle`/`heartbeat_seconds` params)

- [ ] **Step 3: Rewrite `aggregate_stream_to_response` to drive `consume_with_heartbeat`**

```python
from .client_stream import consume_with_heartbeat, _HEARTBEAT, _Timeout

async def aggregate_stream_to_response(
    config, body, *, provider,
    idle_timeout, first_chunk_timeout=None,
    reasoning_idle=None, heartbeat_seconds=10.0, iter_fn=None,
):
    """流式调上游、按 chunk 累加成非流式 dict。超时 → {"_dp_error": ...}。

    走与客户端真流式同一个超时引擎（consume_with_heartbeat），忽略其心跳哨兵、累加
    chunk。reasoning 自适应、first/idle 预算语义与 client_stream 完全一致（单引擎）。
    """
    content_parts, reasoning_parts, tool_calls = [], [], []
    finish_reason, usage = None, None
    fn = iter_fn if iter_fn is not None else iter_litellm_chunks
    upstream = fn(config, body, provider=provider)
    gen = consume_with_heartbeat(
        upstream, idle_timeout=idle_timeout,
        first_chunk_timeout=(first_chunk_timeout or idle_timeout),
        heartbeat_seconds=heartbeat_seconds, log_label="aggregate_stream",
        reasoning_idle=reasoning_idle,
    )
    try:
        async for item in gen:
            if item is _HEARTBEAT:
                continue
            if isinstance(item, _Timeout):
                return {"_dp_error": f"{item.phase} timeout after {item.seconds}s"}
            chunk = item
            if is_error_frame(chunk):
                err = chunk["error"]
                msg = err.get("message") if isinstance(err, dict) else None
                return {"_dp_error": msg or str(err)}
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                r = delta.get("reasoning_content")
                if isinstance(r, str):
                    reasoning_parts.append(r)
                tcs = delta.get("tool_calls")
                if isinstance(tcs, list) and tcs:
                    tool_calls = merge_tool_call_deltas(tool_calls, tcs)
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr
    finally:
        await gen.aclose()

    message = {"role": "assistant"}
    content_text = "".join(content_parts)
    if tool_calls:
        message["content"] = content_text or None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = content_text
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    response = {"choices": [{
        "message": message,
        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
        "index": 0,
    }]}
    if usage is not None:
        response["usage"] = usage
    return response
```

Delete the old `wait_for` loop, the `compute_reasoning_idle` import/use, `reasoning_seen`/`_effective_idle`/`start`/`chunk_count` locals. Remove the now-unused `import time`. Update `stream_aggregated_call` to forward `reasoning_idle`/`heartbeat_seconds` (add params with defaults).

- [ ] **Step 4: Repoint cc timeout reads to `StreamingConfig`**

In `executor.py::execute_consult` (the `aggregate_stream_to_response` call): replace `idle_timeout=float(cc_config.call_timeout_seconds)` / `first_chunk_timeout=float(cc_config.first_chunk_timeout_seconds)` with the `StreamingConfig` values. `execute_consult` receives `config` (ProxyConfig) — use `config.streaming`:

```python
        result = await aggregate_stream_to_response(
            config, consult_body, provider=target_provider,
            idle_timeout=float(config.streaming.idle_timeout_seconds),
            reasoning_idle=float(config.streaming.reasoning_idle_timeout_seconds),
            first_chunk_timeout=float(config.streaming.first_chunk_timeout_seconds),
            heartbeat_seconds=float(config.streaming.heartbeat_seconds),
        )
```

In `router.py` non-stream cc path (~502-516, 526-530): replace `cc_idle = float(self.config.cross_consult.call_timeout_seconds)` / `cc_first = ...` with `sc = self.config.streaming; cc_idle = float(sc.idle_timeout_seconds); cc_first = float(sc.first_chunk_timeout_seconds)` and pass `reasoning_idle=float(sc.reasoning_idle_timeout_seconds), heartbeat_seconds=float(sc.heartbeat_seconds)` to `aggregate_stream_to_response` / `stream_aggregated_call`.

In `stream_cross_consult_continuation` (client_stream.py): the consult call's `with_heartbeat(..., heartbeat_seconds=hb)` — set `hb = float(config.streaming.heartbeat_seconds)`; delete the now-unused `idle = float(cc_config.call_timeout_seconds)` / `first = ...` bindings.

- [ ] **Step 5: Remove timeout fields from `CrossConsultConfig`**

In `deep_proxy/cross_consult/config.py`, delete `call_timeout_seconds`, `first_chunk_timeout_seconds`, `stream_heartbeat_seconds` Fields. Update `tests/test_cross_consult_config.py` (delete `assert c.call_timeout_seconds == 60` and any other removed-field assertions). Update `config.example.yaml`: remove the three cc timeout lines (lines ~84-85 + any cc first_chunk).

- [ ] **Step 6: Run the cc + streaming + config suites**

Run: `python -m pytest tests/test_cross_consult_streaming.py tests/test_cross_consult_config.py tests/test_stream_retry.py tests/test_cross_consult_client_stream.py -v`
Expected: PASS. Fix any test that set `cross_consult.call_timeout_seconds`/`first_chunk_timeout_seconds`/`stream_heartbeat_seconds` to instead set `streaming.*` (grep tests for those field names first).

- [ ] **Step 7: Commit**

```bash
git add deep_proxy/cross_consult/config.py deep_proxy/cross_consult/streaming.py deep_proxy/cross_consult/executor.py deep_proxy/router.py deep_proxy/cross_consult/client_stream.py config.example.yaml tests/
git commit -m "refactor(cc): 合并到单一 StreamingConfig 超时；aggregate 走统一引擎"
```

---

### Task 7: Review + dead-code / stale-comment cleanup + full suite

**Files:**
- Modify: `deep_proxy/cross_consult/reasoning_idle.py` (remove `compute_reasoning_idle` if unused)
- Modify: docstrings/comments across `client_stream.py`, `streaming.py`, `config.py`
- Test: full suite

- [ ] **Step 1: Grep for now-dead symbols**

Run:
```bash
grep -rn "compute_reasoning_idle\|_resolve_idle\|max_stream_total_seconds\|_make_reasoning_aware_idle\|call_timeout_seconds\|stream_heartbeat_seconds" deep_proxy/ --include=*.py
```
Expected: `compute_reasoning_idle` only in `reasoning_idle.py` (def) if all callers removed; `_resolve_idle` none; `max_stream_total_seconds` none; cc `call_timeout_seconds`/`stream_heartbeat_seconds` none. Any remaining production reference → resolve before deleting.

- [ ] **Step 2: Delete `compute_reasoning_idle` if dead**

If Step 1 shows no callers, remove `compute_reasoning_idle` from `reasoning_idle.py` (keep `chunk_has_reasoning`). Update the module docstring (drop the "公式" paragraph).

- [ ] **Step 3: Fix stale comments / docstrings**

Update the "刻意分叉，勿合并" docstrings in `streaming.py` (module + `aggregate_stream_to_response`) and `client_stream.py` (`consume_with_heartbeat`) — they now describe a SINGLE shared engine, not two divergent ones. Update any `idle_ref`/`reasoning_idle_val`/`max_stream_total`/`call_timeout` mentions in comments to match the new single-config, count-based, single-engine reality. Update `StreamingConfig` docstring example if it referenced the old fields.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Final grep for stale references in comments**

Run:
```bash
grep -rn "max_stream_total\|刻意分叉\|两套\|budget 钳\|clamp\|remaining" deep_proxy/cross_consult/ deep_proxy/config.py --include=*.py
```
Fix any comment still describing the removed clamp/deadline/two-engine model.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(timeout): 清理死代码（compute_reasoning_idle 等）+ 过时注释（单引擎/单配置/count-based）"
```

---

## Self-Review notes

- **Coverage:** Step-1 goal (count-based, no clamp, healthy never walled) = Tasks 1,4,5. Step-2 goal (unify engines) = Tasks 2,3,6(aggregate); (merge config) = Task 6. Review+cleanup = Task 7. The four diagnosed incorrectnesses: false-wall → Task 4 (drop clamp); inert cc reasoning → Task 6 (single config, idle 15 < reasoning 45); 45/60/120 divergence → Task 6 (one config) + Task 2 (one engine); misleading `max_stream_total` semantics → Task 1 (removed). Over-engineering: two engines → Tasks 2,3,6; two configs → Task 6; clamp machinery → Task 4; mutable idle_ref → Task 2; knob count → Tasks 1,6.
- **Type consistency:** `consume_with_heartbeat(..., reasoning_idle=None)` (Task 2) consumed by Task 3 + Task 6. `stream_turn_with_retry(make_attempt, *, max_retries, on_result)` + `make_attempt(turn)` single-arg (Tasks 4,5). `aggregate_stream_to_response(..., reasoning_idle, heartbeat_seconds)` (Task 6) called by executor/router/stream_aggregated_call (Task 6 Step 4). `StreamingConfig.max_retries` (Task 1) read in Tasks 4,5,6.
- **Gated deletion:** Task 6 removes cc fields after repointing callers (Step 4 before Step 5); Task 7 deletes `compute_reasoning_idle` only after grep confirms no callers.
- **Ordering note:** Task 5 Step 1 test sets BOTH `cc.first_chunk_timeout_seconds` and `streaming.*` so it passes before AND after Task 6 repoints the cc reads — avoids a transient red between tasks.
