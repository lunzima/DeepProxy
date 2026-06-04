# cross_consult adopts the retry + hard-error timeout model

**Date:** 2026-06-04
**Status:** Design approved, pending implementation plan
**Depends on:** `2026-06-04-mid-stream-timeout-retry-design.md` (plain-path retry, already shipped)

## Problem

The plain (non-cross_consult) streaming path was migrated from the deprecated
"请重试 content + clean finish_reason=stop" notice to a proxy-side retry +
hard-error-frame model. The **cross_consult path was deliberately left on the old
notice** as a follow-up. This spec migrates it.

The cc path emits the deprecated notice in two source-provider turn sites:

- **Initial turn** — `router.DeepProxyRouter._iter_cc_chunks` → `stream_one_turn`,
  then on `turn.errored && turn.timed_out`: `make_timeout_notice_frames` +
  `STREAM_ERRORED`.
- **Resend turn** — `stream_cross_consult_continuation` loop → `stream_one_turn`,
  same notice block per resend.

Both produce a clean `finish_reason=stop` turn whose content is "请重试", which an
agent cannot act on (clean stop = a successful turn). cc traffic therefore still
dies silently on upstream stalls.

The **consult call itself** (`resolve_consult_tool_call` → `execute_consult`) is
**out of scope**: on timeout it already returns an error string as the
`tool_result` (executor.py:105-112) and the resend continues. It is not a
client-facing hang and needs no retry.

## Approved decisions

- **Budget scope:** per source-turn. Each initial/resend turn gets its own
  `max_stream_total_seconds` envelope (fresh deadline per turn). Total cc
  wall-clock may exceed the budget across turns — accepted (cc is inherently
  multi-call; consistent with cc's existing per-call timeouts).
- **Architecture:** one generalized retry skeleton shared by both paths (not a cc
  fork) — fixes the altitude/fork concern raised in review.
- **Config:** the two new knobs (`reasoning_idle_timeout_seconds`,
  `max_stream_total_seconds`) are read from `StreamingConfig` for cc too; cc keeps
  its own `call_timeout_seconds` / `first_chunk_timeout_seconds` /
  `stream_heartbeat_seconds` for idle / TTFT / heartbeat.

## Design

### 1. Generalize the retry skeleton

Extract today's plain-path `stream_with_retry` into a generic skeleton:

```python
async def stream_turn_with_retry(
    make_attempt,                 # Callable[[TurnResult, float], AsyncIterator[dict]]
    *,
    max_total_seconds: float,
    on_result=None,               # Callable[[TurnResult], None] | None
    now=time.monotonic,
) -> AsyncGenerator[dict, None]:
    deadline = now() + max_total_seconds
    committed = False
    while True:
        remaining = deadline - now()
        if remaining <= 0:
            yield make_hard_error_frame("上游持续无响应，超过 …s 总预算，本轮中断。")
            return
        turn = TurnResult()
        async for frame in make_attempt(turn, remaining):
            if _frame_has_visible_output(frame):
                committed = True
            yield frame
        if not turn.timed_out:
            if on_result is not None:
                on_result(turn)            # hand the winning turn to the caller
            return
        if committed:
            yield make_hard_error_frame("已输出部分内容后上游中断，不可续传，本轮中断。")
            return
        yield _HEARTBEAT
```

`make_attempt(turn, remaining)` builds **one fresh attempt's** frame stream: the
caller wires the turn-streamer (`stream_with_idle_timeout` for plain,
`stream_one_turn` for cc), a fresh upstream iterator, accumulator reset/restore,
and pre-content budget clamping (`min(first_chunk, remaining)`,
`min(reasoning_idle, remaining)`) inside it.

`on_result` fires only on a non-timeout exit (clean success **or** a real upstream
error frame already forwarded). The caller inspects the handed-back `turn`:
`turn.errored` (real error) → stop; else proceed. If `on_result` never fires, the
skeleton hard-errored and the caller returns.

**Plain-path adapter.** `stream_with_retry` keeps its current public signature and
becomes a thin wrapper:

```python
async def stream_with_retry(make_upstream, *, idle_timeout, reasoning_idle,
                            first_chunk_timeout, heartbeat_seconds,
                            max_total_seconds, now=time.monotonic):
    def make_attempt(turn, remaining):
        return stream_with_idle_timeout(
            make_upstream(), result=turn,
            idle_timeout=idle_timeout,
            reasoning_idle=min(reasoning_idle, remaining),
            first_chunk_timeout=min(first_chunk_timeout, remaining),
            heartbeat_seconds=heartbeat_seconds,
        )
    async for frame in stream_turn_with_retry(
        make_attempt, max_total_seconds=max_total_seconds, now=now):
        yield frame
```

All existing `test_stream_retry.py` tests (which patch `stream_with_idle_timeout`
and assert clamping/budget) remain valid.

### 2. cc turn-streamer: `stream_one_turn` gains `reasoning_idle`

Add `reasoning_idle: float | None = None` to `stream_one_turn`, mirroring the
`stream_with_idle_timeout` change: `reasoning_idle_val = max(idle, reasoning_idle)
if reasoning_idle is not None else compute_reasoning_idle(idle, first_chunk)`.
`None` preserves current behavior so existing cc tests pass.

### 3. cc wiring

Both notice sites are replaced by `stream_turn_with_retry`:

**Initial turn (`_iter_cc_chunks`):**

```python
snap = accumulator.snapshot()
captured: dict = {}
def make_attempt(turn, remaining):
    accumulator.restore(snap)          # drop a failed attempt's additions; keep prior turns
    return stream_one_turn(
        iter_litellm_chunks(self.config, body, _accumulator=accumulator, provider=provider),
        turn, tool_name=cc.tool_name,
        idle_timeout=float(cc.call_timeout_seconds),
        reasoning_idle=min(float(sc.reasoning_idle_timeout_seconds), remaining),
        first_chunk_timeout=min(float(cc.first_chunk_timeout_seconds), remaining),
        heartbeat_seconds=float(cc.stream_heartbeat_seconds),
    )
async for frame in stream_turn_with_retry(
        make_attempt, max_total_seconds=float(sc.max_stream_total_seconds),
        on_result=lambda t: captured.__setitem__("turn", t)):
    yield frame
turn = captured.get("turn")
if turn is None or turn.errored:        # hard-errored, or real error already forwarded
    return
if not turn.had_cc_call:
    yield make_terminal_frame(turn.finish_reason, turn.accumulated_tool_calls)
    return
async for frame in stream_cross_consult_continuation(...):
    yield frame
```

**Resend turn (`stream_cross_consult_continuation` loop):** identical pattern per
resend — `snapshot()` before the turn, `make_attempt` restores + builds a fresh
`resend_iter`, `on_result` captures the winning turn. After: `turn is None or
turn.errored` → return; else `turn_tool_calls/turn_content = turn...`; `not
had_cc_call` → terminal frame + return; else next round.

### 4. Accumulator snapshot/restore

cc accumulates across turns into one accumulator; the cache must hold the full
forwarded response (initial 前导 + resend answer). A retry discards only the
**failed attempt's** additions:

```python
def snapshot(self) -> dict:
    return {i: {**s} for i, s in self._slots.items()}
def restore(self, snap: dict) -> None:
    self._slots = {i: {**s} for i, s in snap.items()}
```

`restore` deep-copies the slot dicts so the held snapshot isn't mutated by
subsequent `consume`. Plain path keeps `reset()` (single turn; `reset() ≡
restore({})`).

### 5. Cleanup

- Delete `make_timeout_notice_frames` and `_timeout_notice_text` (no callers after
  migration).
- Remove `STREAM_ERRORED` and its handler branch in `iter_chat_chunks` **iff** a
  full grep shows no remaining references after migration (every dirty exit now
  yields an `is_error_frame`, so `saw_error_frame` is set without the sentinel).
  If any reference survives, leave it and note why.
- Revert the "deprecated, cc still uses it" comments added in commit 559c945
  (`make_timeout_notice_frames` docstring, `_timeout_notice_text` docstring,
  `TurnResult` comment, `_iter_cc_chunks` comment) — now genuinely migrated.
- `make_terminal_frame` stays (still used by the no-cc-call / hard-turn-limit
  terminal exits).

## Testing

New (`tests/test_stream_retry.py` or a cc-focused module + `test_cross_consult_*`):

- initial-turn pre-content stall → retry → success (no notice, no error frame).
- initial-turn post-content stall → hard-error frame.
- resend-turn pre-content stall → retry → success.
- cc per-turn budget exhaustion → hard-error frame.
- snapshot/restore: a retry on turn N preserves turn N-1's accumulated content in
  the cache (assert via `flush_to_cache` spy).
- cc emits no `[DeepProxy]` notice text and no clean `finish_reason=stop` on a
  timeout.

Update / delete obsolete tests:

- delete `test_make_timeout_notice_frames_first_chunk` /
  `test_make_timeout_notice_frames_mid_stream_distinct_text` (helper removed).
- rewrite `test_cc_initial_turn_timeout_emits_graceful_notice`
  (`tests/test_stream_timeout_notice.py`) → assert retry-then-hard-error.
- update the module docstring of `test_stream_timeout_notice.py`.

## Files touched (anticipated)

- `deep_proxy/cross_consult/client_stream.py` — `stream_turn_with_retry` (new),
  `stream_with_retry` → adapter, `stream_one_turn` gains `reasoning_idle`,
  `stream_cross_consult_continuation` resend retry, delete notice helpers.
- `deep_proxy/router.py` — `_iter_cc_chunks` initial-turn retry; remove
  `STREAM_ERRORED` handling + import if dead; revert deprecation comments.
- `deep_proxy/compatibility/reasoning_handler.py` — `snapshot()` / `restore()`.
- `tests/` — new cc retry tests; delete/rewrite obsolete notice tests.
