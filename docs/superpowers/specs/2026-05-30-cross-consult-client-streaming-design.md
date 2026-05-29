# Cross-Consult 客户端真流式 — 设计 spec

**日期**：2026-05-30
**状态**：设计待评审
**关联**：`docs/superpowers/plans/2026-05-28-cross-consult.md`、`docs/mimo_integration.md` §12

## 1. 背景与问题

cross_consult 的内部往返（consult 调对偶 provider、重发原 provider）已在
`33bf02b` 全面流式化，但那是**内部**流式：`aggregate_stream_to_response` /
`stream_aggregated_call` 把上游流聚合回非流式 dict，目的是躲墙钟超时。

面向客户端的流式 endpoint `iter_chat_chunks`（`router.py:492`）在
`cc_active`（cross_consult 启用 + provider + 有 pair）时，会 **buffer 所有
chunk**（`router.py:540` `buffered_chunks.append`），流结束后要么回放全部
buffer、要么 yield 一个 `synthesize_final_stream_chunk` 合成块。

**结果：只要 cross_consult 激活，客户端拿到的"流"其实是憋到最后一次性吐。**
本设计把它改成真流式——边到边推送。

## 2. 范围

- **仅** `iter_chat_chunks`（流式 endpoint）。
- 非流式 `chat_completions`（`router.py:452`）返回 dict、无法 stream，保持
  现有 aggregate-loop 行为不变。
- consult 内部调用 / 重发内部调用仍走流式聚合（不改 `executor.py` /
  `streaming.py` 的内部聚合语义；但重发在客户端路径改为逐 chunk 透传）。

## 3. 已确认的行为决策

| 维度 | 决策 | 含义 |
|---|---|---|
| 静默间隙（consult 执行 + 重发 prefill） | **keep-alive 心跳** | 间隙期周期性发 SSE 注释帧，保持连接温热、防客户端 idle-read 超时 |
| 前导文本（cc 工具调用前的 content） | **透传可见** | content/reasoning 边到边推；客户端会看到「让我咨询…」之类前导 |
| 多段 reasoning（每轮各有 reasoning_content） | **全部透传** | 初始轮 + 每次重发轮的 reasoning 都流给客户端 |

三项一致指向同一模型：**content + reasoning + 终轮 tool_calls 全部透传，
仅抑制 `cross_consult` 工具帧，间隙发心跳，跨重发轮桥接。**

## 4. 架构（Approach 1：流式原生 continuation）

三个隔离单元 + `iter_chat_chunks` 的接线改动。

### 4.1 单元 A：单轮流式器 `stream_one_turn`

**职责**：消费一个上游 turn 的 chunk 流，把面向客户端的部分边到边 yield，
同时累加 tool_calls 留到轮末判定。

**位置**：`deep_proxy/cross_consult/client_stream.py`（新文件）

**签名**（async generator + 显式可变结果容器，避免「生成器耗尽后读属性」的
歧义）：
```python
@dataclass
class TurnResult:
    accumulated_tool_calls: list[dict] = field(default_factory=list)
    content: str = ""            # 累加的 assistant 文本，供重发轮重建消息历史
    had_cc_call: bool = False
    finish_reason: str | None = None
    errored: bool = False

async def stream_one_turn(
    chunk_iter: AsyncIterator[dict],
    result: TurnResult,          # 调用方传入；生成器在轮末就地填充
    *,
    tool_name: str,
    idle_timeout: float,
    first_chunk_timeout: float,
    heartbeat_seconds: float,
) -> AsyncGenerator[dict, None]:
    # yield 客户端帧（content/reasoning delta）或心跳帧 {"_dp_heartbeat": True}
    # 耗尽后 result 字段已填好，调用方在 `async for` 之后读取
```

**规则**（核心）：
- **content / reasoning_content delta**：立即 yield（构造仅含这些字段的客户端
  chunk）。
- **tool_calls delta**：**不**立即 yield，用 `merge_tool_call_deltas`
  （`utils.py`）累加到本轮 buffer。理由：在轮末（finish_reason 到达）前无法
  判定本轮是「cc 轮」（tool_calls 进消息历史、不给客户端）还是「终轮」
  （tool_calls 透传给客户端）。content/reasoning 无此歧义，故照常透传。
- **finish_reason**：暂存，不随中间 chunk 透传（避免提前给客户端「结束」信号）。
- **心跳**：等待下一 chunk 时，每 `heartbeat_seconds` 无 chunk 就 yield 一个
  `{"_dp_heartbeat": True}`；累计等待超过预算（首 chunk 用 first_chunk_timeout、
  之后用 idle_timeout，复用 §previous-fix 的双预算）则视为 hang，结束并标记错误。
- **error frame**（`{"error": {...}}`）：透传给客户端并终止本轮。

**轮末产物**（供 continuation 决策）：`accumulated_tool_calls`、
`had_cc_call`（是否含 name==tool_name 的调用）、`finish_reason`、`errored`。

### 4.2 单元 B：心跳包裹 `with_heartbeat`

**职责**：在一个 awaitable（consult 执行）进行期间周期性 yield 心跳帧。

```python
async def with_heartbeat(
    awaitable: Awaitable[T], *, heartbeat_seconds: float,
) -> AsyncGenerator[dict | _Done[T], None]:
    task = asyncio.create_task(awaitable)
    while True:
        done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
        if task in done:
            yield _Done(task.result()); return
        yield {"_dp_heartbeat": True}
```

continuation 用它包裹 `execute_consult(...)`，把心跳帧 yield 给客户端，最后
拿到 consult 文本结果。（单元 A 的 heartbeat 覆盖重发 prefill / mid-stream
间隙；单元 B 覆盖 consult 执行间隙。两者同一 sentinel。）

### 4.3 单元 C：流式 continuation 循环 `stream_cross_consult_continuation`

**职责**：等价于 `execute_cross_consult_loop`（`interceptor.py:216`）的**流式
变体**——同样的 consult / 消息追加 / 重发逻辑，但 yield 客户端帧而非返回 dict。

**位置**：`client_stream.py`

**入参**：初始轮的 `accumulated_tool_calls`（由 `iter_chat_chunks` 透传初始
流时已累加）、`body`、`source_provider`、`config`、`cc_config`、accumulator。

**循环**（复用 `interceptor._extract_cross_consult_tool_calls` /
`_resolve_consult_tool_call`）：
```
turn_tool_calls = 初始 accumulated_tool_calls
for _turn in range(max_calls*2+1):
    cc_calls = _extract_cross_consult_tool_calls({...turn_tool_calls...}, tool_name)
    if not cc_calls: break        # 本轮无 cc 调用 → 终轮，tool_calls 已在调用方透传
    body.messages.append(assistant_msg(turn_tool_calls))
    for tc in cc_calls:
        async for frame in with_heartbeat(_resolve_consult_tool_call(tc,...)):
            if isinstance(frame, _Done): tool_text, consumed = frame.value
            else: yield frame                      # 心跳
        body.messages.append(tool_result_msg(tc, tool_text))
    # 重发：流式，逐 chunk 透传
    resend_iter = iter_litellm_chunks(config, body, provider=source_provider, _accumulator=acc)
    turn = TurnResult()
    async for frame in stream_one_turn(resend_iter, turn, ...): yield frame  # content/reasoning/心跳/error
    turn_tool_calls = turn.accumulated_tool_calls
    if turn.errored: return
# 终轮：若有非 cc tool_calls（agent 自己的工具），合成 tool_calls 帧 yield 给客户端
yield 终轮的 finish_reason / 非cc tool_calls 帧
```

### 4.4 `iter_chat_chunks` 接线改动

- `cc_active` 分支不再 buffer content：初始流通过 `stream_one_turn` 透传
  （content/reasoning 边到边 yield，tool_calls 累加）。
- 初始轮结束：
  - `had_cc_call == False` → 把初始轮 buffer 的 tool_calls + finish_reason
    作为终轮帧 yield（等价当前「无 cc 直接回放」），结束。
  - `had_cc_call == True` → 交给 `stream_cross_consult_continuation`，把它
    yield 的帧逐个透传，结束。
- 删除 `synthesize_final_stream_chunk` 调用（不再需要合成单块；该 helper
  可保留给非流式 endpoint 或删除——见 §11 待定）。

## 5. 数据流（cc 触发，max_calls≥1）

```
client SSE 流：
  初轮 reasoning delta…  → 初轮 content「让我咨询 mimo…」delta…
  → [consult 执行：每 Ns 一个 : keep-alive]
  → 重发轮 reasoning delta…  → 重发轮 content「综合两者，答案是…」delta…
  → finish_reason=stop → data: [DONE]
（cross_consult 工具帧、初轮 finish_reason=tool_calls 全程被抑制，客户端不可见）
```

## 6. 线缆协议（心跳）

- `iter_chat_chunks` yield 的心跳 sentinel：`{"_dp_heartbeat": True}`（dict，
  保持 iter_chat_chunks 的 dict 流契约）。
- 协议层 `chat_completions_stream`（`router.py:589`）识别该 sentinel，发
  **SSE 注释帧** `: keep-alive\n\n`（**不**做 json dump、**不**加 `data:`
  前缀）。SSE 规范明确忽略 `:` 开头的注释行，零风险污染 delta 解析。
- 普通帧仍走 `data: {json}\n\n`。

## 7. 配置

`CrossConsultConfig`（`cross_consult/config.py`）新增：
```python
stream_heartbeat_seconds: int = Field(
    default=10, ge=1, le=120,
    description="客户端真流式下，静默间隙（consult 执行 / 重发 prefill）期间发送 "
                "SSE keep-alive 注释帧的间隔秒数。须显著小于客户端 idle-read 超时。",
)
```
`config.example.yaml` 同步注释。复用既有 `call_timeout_seconds`（inter-chunk
idle）与 `first_chunk_timeout_seconds`（首 chunk 预算）作为单元 A 的等待预算。

## 8. 缓存 / 记账

- 重发轮的 `iter_litellm_chunks` 复用初始的 `StreamingReasoningAccumulator`
  （`_accumulator=acc`），使 accumulator 看到全部 turn 的 content/reasoning →
  `finally` 块的 `flush_to_cache` 反映完整多轮终态。
- cc 工具帧不透传、不入 accumulator（与「客户端从不可见 cc 调用」一致）。
- `_commit_pending_upgrade`：维持 `finally` 中 `completed_cleanly and not
  saw_error_frame` 的现有门控；`stream_one_turn` 的 error/超时须设
  `saw_error_frame=True`。
- 逐 delta 的 reasoning 规整已在 `iter_litellm_chunks` 内
  （`process_streaming_delta`）完成，替代了原 dict 级 `process_response`
  对重发响应的处理；spec 实现时须验证 parity（reasoning 抽取、null 清理）。

## 9. 错误处理

- consult 失败：`_resolve_consult_tool_call` / `execute_consult` 已返回
  错误前缀字符串作为 tool_result，循环继续（与现有非流式 loop 一致）。
- 重发轮 error frame 或超时（单元 A 标记 errored）：透传 error frame（若有）
  并终止 continuation，设 `saw_error_frame`。
- 硬轮次上限 `max_calls*2+1` 保留（防无限循环）。

## 10. 测试

**单元（`tests/test_cross_consult_client_stream.py`，mock chunk 流，快）**：
- `stream_one_turn`：content/reasoning 即时透传；tool_calls 不透传、轮末
  正确累加；`had_cc_call` 判定正确；finish_reason 暂存。
- `stream_one_turn` 抑制 cc 工具帧 + 初轮 finish_reason=tool_calls。
- `stream_one_turn` 心跳：等待超 heartbeat_seconds 无 chunk → yield 心跳；
  超双预算 → errored。
- `with_heartbeat`：慢 awaitable → 周期心跳 + 最终结果。
- 协议层：`{"_dp_heartbeat": True}` → `: keep-alive\n\n`（非 data 帧）。

**集成（扩 `tests/test_cross_consult_loop.py`）**：
- 改写 `test_iter_chat_chunks_intercepts_cross_consult_in_stream`：现在断言
  初始 content + 重发 content 都逐帧到达，而非单一合成块。
- 前导文本透传可见。
- 多段 reasoning（初轮 + 重发轮）都到达。
- consult 间隙有心跳帧。
- cc 工具帧客户端不可见。
- 无 cc 调用时行为与现状一致（content 透传 + 终轮 tool_calls/finish_reason）。
- 重发超时 → error frame + 不提交升格记账。

## 11. 不做（YAGNI / out of scope）

- 不改非流式 `chat_completions` 的 aggregate-loop。
- 不把 consult 的 reasoning 作为「可见进度」转发（已选 keep-alive 而非可见提示）。
- 不改 redirect（整轮重定向）路径。
- `synthesize_final_stream_chunk` 是否删除：流式路径不再用；保留与否在实现
  PR 决定（若无其它引用则删）。

## 12. 单元边界自检

- `stream_one_turn`：输入 chunk 流 + 预算/工具名 → 输出客户端帧流 + 轮末元数据。
  不依赖 router 状态，可独立测。
- `with_heartbeat`：输入 awaitable + 间隔 → 心跳帧流 + 结果。纯通用，可独立测。
- `stream_cross_consult_continuation`：编排上两者 + 复用 interceptor 的
  consult/验证 helper；依赖注入 body/config/provider/accumulator，可测。
- 协议层心跳序列化：隔离在 `chat_completions_stream`，单点。
