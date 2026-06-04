# Mid-stream timeout → proxy retry loop + hard error

**Date:** 2026-06-04
**Status:** Design approved, pending implementation plan
**Scope:** Plain (non-cross_consult) streaming path

## Problem

When the upstream stalls mid-stream, the plain streaming path emits a "请直接重试本次请求以继续" notice and stops. The user has confirmed this **always causes the agent to stop working** and **never produces a retry**.

### Root cause

On a plain-path mid-stream stall, `stream_with_idle_timeout`
(`deep_proxy/cross_consult/client_stream.py:397`) injects two frames via
`make_timeout_notice_frames`:

1. a `content` delta carrying the "请直接重试…" text, then
2. a clean `finish_reason=stop` terminal frame,

then returns. The `STREAM_ERRORED` sentinel that `_iter_plain_chunks` yields
afterward is **swallowed** at `router.py:592` (`continue` — never forwarded to
the client).

So the client SDK receives exactly one **clean, successful** turn whose answer
happens to be the words "please retry", followed by `[DONE]`. From the protocol's
view the turn completed normally (`finish_reason=stop`). An autonomous agent
(Claude Code, OpenAI SDK consumers) has no mechanism that reads a natural-language
"please retry" and acts on it — it renders the message and ends the turn. The
original design premise ("agent reads the notice and decides to retry") is
**structurally impossible**.

**The fix:** stop asking the agent to retry. Make the proxy itself retry on
recoverable (pre-content) stalls, and surface a **real error** (not a clean stop)
only when genuinely exhausted, so the SDK raises instead of silently completing.

## Behavior model

Three phases, distinguished by whether any **client-visible** token
(`content` or `tool_calls` delta — `reasoning_content` does **not** count) has
been forwarded:

| Phase | Stall detection budget | On stall |
|---|---|---|
| **Before first chunk** (TTFT / prefill) | `first_chunk_timeout_seconds` (120s) | pre-content → **retry fresh** |
| **Reasoning streaming, no content yet** | `reasoning_idle_timeout_seconds` (45s) | pre-content → **retry fresh** |
| **Content / tool_calls streaming** | `idle_timeout_seconds` (15s) | post-content → **hard error immediately** |

Wrapping all phases: an overall wall-clock budget `max_stream_total_seconds`
(600s). Pre-content retries repeat until the upstream completes the turn **or**
the 600s budget is exhausted → **hard error**.

Heartbeats (`heartbeat_seconds`, 10s) flow continuously — within each attempt and
in the gap between an abandoned attempt and the next fresh attempt — so the client
connection (whose read timeout must exceed the heartbeat interval) stays warm
across the entire 600s envelope.

### Rationale for the per-phase budgets

- **First-chunk 120s (unchanged):** max-reasoning prefill legitimately takes tens
  of seconds before the first token; an aggressive value would retry healthy-but-slow
  prefills. It is still pre-content, so a genuine timeout retries fresh.
- **Reasoning 45s (new, separate):** V4 deep reasoning can have multi-second
  thinking pauses. A separate, larger window than the content idle tolerates normal
  pauses and only retries a genuinely dead reasoning stream — avoiding wasted
  upstream reasoning tokens. Replaces the old reasoning-aware bump that raised idle
  all the way to `first_chunk_timeout` (120s).
- **Content 15s ("retry harder"):** once visible output is flowing, a 15s gap is a
  strong stall signal. Post-content is non-recoverable (no prefix-continuation), so
  this triggers an immediate hard error rather than a retry.

## Architecture

### `stream_with_idle_timeout` → detection only

Today it both detects the timeout **and** injects the notice/clean-stop frames.
The injection is removed. On timeout it sets `TurnResult.timed_out` /
`TurnResult.timeout_phase` / `TurnResult.timeout_seconds` (as it already does) and
simply returns. **Policy** (retry vs. hard error) moves to the caller.

The reasoning-aware idle helper changes signature from
`_make_reasoning_aware_idle(idle_timeout, first_chunk_timeout)` to
`_make_reasoning_aware_idle(idle_timeout, reasoning_idle)` — once
`reasoning_content` is seen, the effective idle bumps to
`reasoning_idle_timeout_seconds` (45s) instead of `first_chunk_timeout` (120s).

### `_iter_plain_chunks` → retry loop

`_iter_plain_chunks` (`router.py:651`) becomes the policy layer:

```
deadline   = monotonic() + max_stream_total_seconds      # 600s envelope
committed  = False                                       # any visible content/tool_call forwarded?

loop:
    attempt = stream_with_idle_timeout(
        fresh iter_litellm_chunks(config, body, ...),     # NEW upstream call each attempt
        result=turn,
        idle_timeout=idle_timeout_seconds,                # 15s content idle
        reasoning_idle=reasoning_idle_timeout_seconds,    # 45s reasoning idle
        first_chunk_timeout=first_chunk_timeout_seconds,  # 120s
        heartbeat_seconds=heartbeat_seconds,
    )
    for chunk in attempt:
        if chunk carries content/tool_calls delta: committed = True
        yield chunk                                       # reasoning/content/tool_calls/finish/heartbeat

    if turn finished via upstream finish_reason:          # success
        return
    if turn.timed_out:
        if committed:               yield HARD_ERROR; return    # post-content
        if monotonic() >= deadline: yield HARD_ERROR; return    # budget exhausted
        yield heartbeat; continue                                # pre-content → retry fresh
    if real upstream error frame already forwarded:       # {"error": {...}} from upstream
        return                                            # already a hard error, no retry
```

Notes:
- **Fresh upstream call per attempt.** Each retry constructs a new
  `iter_litellm_chunks(config, body, ...)` over the unmodified original `body`. A
  pre-content retry re-streams reasoning from scratch; since no visible content was
  sent, the client only saw heartbeats, so this is seamless. (Reasoning_content
  already streamed in a prior attempt may re-appear; accepted — reasoning is not the
  answer and agents tolerate restarted thinking. Reasoning dedup is explicitly out
  of scope / YAGNI.)
- **`committed` latches True** on the first visible token and never resets, so a
  retry can only ever happen before any content is shown. Once content begins, the
  next stall is a hard error.
- Use `time.monotonic()` for the deadline (not wall-clock).

### Hard error frame

A **client-visible** error frame, distinct from the swallowed `STREAM_ERRORED`
sentinel:

```json
{"error": {"message": "<honest reason>", "type": "timeout_error", "code": 504}}
```

`is_error_frame` already routes this: `chat_completions_stream` forwards the frame
as `data: {...}` then emits `[DONE]` and stops (`router.py:691`). The SDK sees an
error object in the stream and **raises** — the desired hard failure, instead of a
spuriously successful turn.

Honest messages, e.g.:
- budget exhausted: `"上游持续无响应，已重试 N 次仍超过 600s 预算，本轮中断。"`
- post-content stall: `"已输出部分内容后上游中断，不可续传，本轮中断。"`

The plain-path use of `make_timeout_notice_frames` (`client_stream.py:397`) and the
"请直接重试" content notice are **removed**. The notice/clean-stop premise is the bug.

## Configuration (`StreamingConfig`, `deep_proxy/config.py:525`)

| Field | Old | New | Note |
|---|---|---|---|
| `idle_timeout_seconds` | 60 | **15** | content-phase stall window |
| `reasoning_idle_timeout_seconds` | — | **45** | new; effective idle once reasoning seen |
| `first_chunk_timeout_seconds` | 120 | 120 | unchanged |
| `max_stream_total_seconds` | — | **600** | new; overall wall-clock budget, exhaustion → hard error |
| `heartbeat_seconds` | 10 | 10 | unchanged; must stay < client read timeout |

The `StreamingConfig` docstring's "超时不报错…注入 assistant content + clean finish"
paragraph is rewritten to describe the retry-loop + hard-error contract.

## Scope & non-goals

- **Plain path only.** The reported message originates here
  (`cross_consult` defaults disabled). The cross_consult path
  (`make_timeout_notice_frames` at `client_stream.py:507` and `router.py:635`)
  retains its current behavior — a separate follow-up. `make_timeout_notice_frames`
  / `_timeout_notice_text` are **not deleted**; only the plain-path caller stops
  using them.
- **No prefix-continuation** (explicit user decision). Post-content stalls are
  non-recoverable by design.
- **Real upstream `{"error"}` frames forward as-is** with no retry — only *timeouts*
  drive the retry loop.
- **No reasoning dedup** across retries.

## Testing

New / changed tests (`tests/test_cross_consult_client_stream.py` and a plain-path
suite):

- **retry-fresh on pre-content stall:** first attempt stalls during reasoning,
  second attempt streams content + finish → client sees a clean successful turn,
  no error frame.
- **hard error on post-content stall:** attempt streams content then stalls →
  client receives an `{"error"}` frame + `[DONE]`, no clean `finish_reason=stop`.
- **hard error on budget exhaustion:** every attempt stalls pre-content until
  `max_stream_total_seconds` elapses → `{"error"}` frame.
- **heartbeats across retries:** heartbeat frames emitted within and between
  attempts.
- **update `test_stream_with_idle_timeout_emits_notice_on_timeout`:** the notice
  frames no longer exist; assert detection-only return + `TurnResult.timed_out`.
- **first-chunk pre-content retry:** no chunk within `first_chunk_timeout` → retry
  fresh (not hard error) while within budget.

## Files touched (anticipated)

- `deep_proxy/config.py` — `StreamingConfig`: new fields, docstring.
- `deep_proxy/cross_consult/client_stream.py` — `stream_with_idle_timeout`
  detection-only; `_make_reasoning_aware_idle` signature; keep
  `make_timeout_notice_frames` for cc path.
- `deep_proxy/router.py` — `_iter_plain_chunks` retry loop; hard-error frame
  construction; pass new config values.
- `tests/test_cross_consult_client_stream.py` (+ new plain-path tests).
