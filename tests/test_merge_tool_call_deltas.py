"""utils.merge_tool_call_deltas canonical 语义回归测试。

OpenAI 流式 tool_calls 累加的"覆盖 vs 拼接"是项目内多处依赖的不变量
（router.py iter_chat_chunks / reasoning_handler 的 StreamingReasoningAccumulator）。
本测试钉住语义防止悄无声息的回退：
  - id / type / function.name 覆盖（= 语义）
  - function.arguments 拼接（+= 语义）
"""
from __future__ import annotations

from deep_proxy.utils import merge_tool_call_deltas


def test_arguments_concatenate_across_chunks():
    """arguments JSON 字符串按 chunk 拼接（OpenAI 流式约定）。"""
    chunk1 = [{"index": 0, "id": "tc1", "type": "function",
               "function": {"name": "get_weather", "arguments": ""}}]
    chunk2 = [{"index": 0, "function": {"arguments": '{"city": '}}]
    chunk3 = [{"index": 0, "function": {"arguments": '"Shanghai"}'}}]

    result = merge_tool_call_deltas([], chunk1)
    result = merge_tool_call_deltas(result, chunk2)
    result = merge_tool_call_deltas(result, chunk3)

    assert len(result) == 1
    assert result[0]["function"]["arguments"] == '{"city": "Shanghai"}'


def test_name_overwrite_not_concat():
    """function.name 必须覆盖而非拼接 —— 若上游重复发 name 不应产生 'getget'。

    这是 canonical 语义关键点：与 router.py iter_chat_chunks 旧实现的 `+=` 行为分歧。
    """
    chunk1 = [{"index": 0, "id": "tc1", "type": "function",
               "function": {"name": "get_weather", "arguments": ""}}]
    # 上游重发 name（一些 provider 在 stream 中段重复给）
    chunk2 = [{"index": 0, "function": {"name": "get_weather"}}]

    result = merge_tool_call_deltas([], chunk1)
    result = merge_tool_call_deltas(result, chunk2)

    assert result[0]["function"]["name"] == "get_weather", (
        f"name 必须覆盖而非拼接，实际得到 {result[0]['function']['name']!r}"
    )


def test_id_preserved_when_subsequent_chunks_omit_it():
    """id 在首 chunk 给定后，后续 chunk 即使不带 id 也保持。"""
    chunk1 = [{"index": 0, "id": "tc_xyz", "type": "function",
               "function": {"name": "f", "arguments": ""}}]
    chunk2 = [{"index": 0, "function": {"arguments": "abc"}}]

    result = merge_tool_call_deltas([], chunk1)
    result = merge_tool_call_deltas(result, chunk2)

    assert result[0]["id"] == "tc_xyz"


def test_multiple_tool_calls_by_index():
    """多个并行 tool_call 按 index 分别累加，互不干扰。"""
    chunk1 = [
        {"index": 0, "id": "tc0", "type": "function",
         "function": {"name": "f0", "arguments": ""}},
        {"index": 1, "id": "tc1", "type": "function",
         "function": {"name": "f1", "arguments": ""}},
    ]
    chunk2 = [
        {"index": 1, "function": {"arguments": '{"x":'}},
        {"index": 0, "function": {"arguments": '{"y":'}},
    ]
    chunk3 = [
        {"index": 0, "function": {"arguments": "1}"}},
        {"index": 1, "function": {"arguments": "2}"}},
    ]

    result = merge_tool_call_deltas([], chunk1)
    result = merge_tool_call_deltas(result, chunk2)
    result = merge_tool_call_deltas(result, chunk3)

    assert len(result) == 2
    # 输出按 index 升序排列
    assert result[0]["index"] == 0
    assert result[1]["index"] == 1
    assert result[0]["function"]["name"] == "f0"
    assert result[0]["function"]["arguments"] == '{"y":1}'
    assert result[1]["function"]["name"] == "f1"
    assert result[1]["function"]["arguments"] == '{"x":2}'


def test_new_tool_call_appears_mid_stream():
    """中段出现新的 index → 累加器为其创建空 slot。"""
    chunk1 = [{"index": 0, "id": "a", "type": "function",
               "function": {"name": "first", "arguments": "{}"}}]
    # 中段冒出 index=1
    chunk2 = [{"index": 1, "id": "b", "type": "function",
               "function": {"name": "second", "arguments": ""}}]

    result = merge_tool_call_deltas([], chunk1)
    result = merge_tool_call_deltas(result, chunk2)

    assert len(result) == 2
    assert {r["function"]["name"] for r in result} == {"first", "second"}


def test_default_slot_has_function_dict():
    """无 function 字段的 delta 不应崩溃；空 slot 初始化合理默认值。"""
    # 只发 id，不发 function
    chunk1 = [{"index": 0, "id": "tc1"}]
    result = merge_tool_call_deltas([], chunk1)

    assert result[0]["id"] == "tc1"
    assert result[0]["function"] == {"name": "", "arguments": ""}
    assert result[0]["type"] == "function"


def test_empty_deltas_returns_existing_unchanged():
    existing = [{"index": 0, "id": "tc", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}]
    result = merge_tool_call_deltas(existing, [])
    assert result == [existing[0]]
