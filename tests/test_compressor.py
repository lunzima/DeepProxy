"""测试 LLM-based system prompt 压缩器（带磁盘缓存）。

不真实调上游 —— 替换 _call_llm 为 mock 验证缓存逻辑。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deep_proxy.optimization.compressor import (
    _CACHE_VERSION,
    SystemPromptCompressor,
)


def _make_compressor(tmp_path: Path, **kwargs) -> SystemPromptCompressor:
    return SystemPromptCompressor(
        cache_path=tmp_path / "cache.json",
        api_key="sk-test",
        api_base="https://api.test/v1",
        **kwargs,
    )


class TestCompressorCaching:
    async def test_first_call_returns_original_and_schedules_background(self, tmp_path):
        """非阻塞：首次调用立即返回原文，压缩在后台跑。"""
        c = _make_compressor(tmp_path)
        calls = {"n": 0}

        async def fake_llm(text):
            calls["n"] += 1
            return f"[compressed] {text[:20]}"

        c._call_llm = fake_llm  # type: ignore[assignment]
        original = "Hello world this is a long system prompt."
        out = await c.compress(original)
        # 主路径返回原文（非阻塞）
        assert out == original
        # 等后台压缩完成
        await c.wait_inflight()
        assert calls["n"] == 1

    async def test_second_call_hits_memory_after_background_completes(self, tmp_path):
        c = _make_compressor(tmp_path)
        calls = {"n": 0}

        async def fake_llm(text):
            calls["n"] += 1
            return "X"

        c._call_llm = fake_llm  # type: ignore[assignment]
        # 第 1 次 → 返回原文 + 调度后台
        first = await c.compress("same text")
        assert first == "same text"
        await c.wait_inflight()  # 等后台 done
        # 第 2/3 次 → 命中内存缓存
        assert await c.compress("same text") == "X"
        assert await c.compress("same text") == "X"
        assert calls["n"] == 1

    async def test_concurrent_same_key_only_one_background_task(self, tmp_path):
        """同一 key 并发 compress() 只调度一个后台任务。"""
        c = _make_compressor(tmp_path)
        calls = {"n": 0}

        async def fake_llm(text):
            calls["n"] += 1
            await asyncio.sleep(0.02)
            return f"compressed-{calls['n']}"

        c._call_llm = fake_llm  # type: ignore[assignment]
        # 并发触发，全部应立即返回原文
        results = await asyncio.gather(
            c.compress("text"),
            c.compress("text"),
            c.compress("text"),
        )
        assert all(r == "text" for r in results)
        await c.wait_inflight()
        assert calls["n"] == 1  # 只调度 1 次后台任务

    async def test_persisted_to_disk_after_background(self, tmp_path):
        c = _make_compressor(tmp_path)

        async def fake_llm(text):
            return "PERSISTED"

        c._call_llm = fake_llm  # type: ignore[assignment]
        await c.compress("hello")
        await c.wait_inflight()

        cache_file = tmp_path / "cache.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert data["version"] == _CACHE_VERSION
        assert len(data["entries"]) == 1
        assert "PERSISTED" in next(iter(data["entries"].values()))

    async def test_loaded_from_disk_on_init(self, tmp_path):
        # 先创建一个 compressor 写一条
        c1 = _make_compressor(tmp_path)

        async def fake1(text):
            return "FROM_DISK"

        c1._call_llm = fake1  # type: ignore[assignment]
        await c1.compress("hello")
        await c1.wait_inflight()
        # 新 compressor 实例加载磁盘
        c2 = _make_compressor(tmp_path)
        calls = {"n": 0}

        async def should_not_be_called(text):
            calls["n"] += 1
            return "BAD"

        c2._call_llm = should_not_be_called  # type: ignore[assignment]
        out = await c2.compress("hello")
        assert out == "FROM_DISK"  # 第二次实例直接命中
        assert calls["n"] == 0

    async def test_cache_version_mismatch_ignored(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(
            json.dumps({"version": _CACHE_VERSION + 999, "entries": {"x": "OLD"}}),
            encoding="utf-8",
        )
        c = _make_compressor(tmp_path)
        # 旧条目应被忽略
        assert len(c._mem) == 0


class TestCompressorFailureModes:
    async def test_llm_failure_returns_original_no_cache_write(self, tmp_path):
        c = _make_compressor(tmp_path)

        async def boom(text):
            raise RuntimeError("upstream down")

        c._call_llm = boom  # type: ignore[assignment]
        original = "important system prompt"
        out = await c.compress(original)
        assert out == original  # fail-fast 立即返回原文
        await c.wait_inflight()  # 等后台失败完成
        # 后台失败 → 不写缓存
        assert (tmp_path / "cache.json").exists() is False

    async def test_empty_input_passthrough(self, tmp_path):
        c = _make_compressor(tmp_path)

        async def must_not_call(text):
            raise AssertionError("should not invoke LLM for empty input")

        c._call_llm = must_not_call  # type: ignore[assignment]
        assert await c.compress("") == ""
        assert await c.compress("   \n  ") == "   \n  "

    async def test_empty_compression_does_not_cache(self, tmp_path):
        c = _make_compressor(tmp_path)

        async def returns_blank(text):
            return "   "

        c._call_llm = returns_blank  # type: ignore[assignment]
        original = "x" * 100
        out = await c.compress(original)
        assert out == original
        await c.wait_inflight()
        # 不应写缓存
        assert original not in c._mem.values() if c._mem else True
        assert len(c._mem) == 0

    async def test_failed_background_allows_retry_next_request(self, tmp_path):
        """后台压缩失败后，再次 compress() 应该重新调度（in-flight 槽已释放）。"""
        c = _make_compressor(tmp_path)
        attempts = []

        async def maybe_succeed(text):
            attempts.append(text)
            if len(attempts) == 1:
                raise RuntimeError("first try fails")
            return "SUCCESS"

        c._call_llm = maybe_succeed  # type: ignore[assignment]
        await c.compress("test")
        await c.wait_inflight()  # 第一次失败
        assert len(attempts) == 1
        await c.compress("test")
        await c.wait_inflight()  # 第二次成功
        assert len(attempts) == 2
        assert "SUCCESS" in c._mem.values()


class TestCompressedCombinedSystem:
    """验证 apply_cheap_optimizations 会把 skills + 用户 system 合并后整体送压缩。

    非阻塞行为：首次请求传输原文 + 后台调度，后续请求命中缓存。
    """

    async def test_combines_user_system_with_skills_for_compression(self, tmp_path):
        from deep_proxy.optimization import apply_cheap_optimizations

        c = _make_compressor(tmp_path)
        seen_inputs = []

        async def fake_llm(text):
            seen_inputs.append(text)
            return "<COMPRESSED>"

        c._call_llm = fake_llm  # type: ignore[assignment]

        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for code review."},
                {"role": "user", "content": "review this"},
            ],
        }
        await apply_cheap_optimizations(
            body,
            avoid_negative_style=True, natural_temperament=False, contextual_register=False,
            assume_good_intent=False,
            instruction_priority=False, independent_analysis=False, inject_date=False,
            show_math_steps=False, prefer_multiple_sources=False,
            avoid_fabricated_citations=False, json_mode_hint=False,
            safe_inlined_content=False, re2=False, cot_reflection=False,
            readurls=False, compressor=c,
        )
        await c.wait_inflight()  # 等后台压缩完成
        # 压缩器看到的输入应同时含 skill 文本和用户 system 文本
        assert len(seen_inputs) == 1
        combined = seen_inputs[0]
        assert "抚慰套话" in combined  # skill
        assert "code review" in combined  # 用户 system
        # 首次请求：消息中是原文（非阻塞），不是压缩版
        assert body["messages"][0]["role"] == "system"
        assert "抚慰套话" in body["messages"][0]["content"]
        assert "code review" in body["messages"][0]["content"]

    async def test_second_request_uses_compressed_after_background(self, tmp_path):
        """首次请求传原文，等后台压缩完成；再次相同请求即用压缩版。"""
        from deep_proxy.optimization import apply_cheap_optimizations

        c = _make_compressor(tmp_path)

        async def fake_llm(text):
            return "<COMPRESSED>"

        c._call_llm = fake_llm  # type: ignore[assignment]

        def make_body():
            return {
                "messages": [
                    {"role": "system", "content": "Be a code reviewer."},
                    {"role": "user", "content": "review"},
                ],
            }

        kwargs = dict(
            avoid_negative_style=True, natural_temperament=False, contextual_register=False,
            assume_good_intent=False, instruction_priority=False,
            independent_analysis=False, inject_date=False,
            show_math_steps=False, prefer_multiple_sources=False,
            avoid_fabricated_citations=False, json_mode_hint=False,
            safe_inlined_content=False, re2=False, cot_reflection=False,
            readurls=False, compressor=c,
        )
        # 第 1 次：传输原文
        b1 = make_body()
        await apply_cheap_optimizations(b1, **kwargs)
        assert "抚慰套话" in b1["messages"][0]["content"]
        await c.wait_inflight()
        # 第 2 次：命中缓存，传输压缩版
        b2 = make_body()
        await apply_cheap_optimizations(b2, **kwargs)
        assert b2["messages"][0]["content"] == "<COMPRESSED>"

    async def test_no_user_system_compresses_skills_only(self, tmp_path):
        from deep_proxy.optimization import apply_cheap_optimizations

        c = _make_compressor(tmp_path)
        seen = []

        async def fake_llm(text):
            seen.append(text)
            return "C"

        c._call_llm = fake_llm  # type: ignore[assignment]

        body = {"messages": [{"role": "user", "content": "hi"}]}
        await apply_cheap_optimizations(
            body, avoid_negative_style=True, contextual_register=False,
            assume_good_intent=False, instruction_priority=False,
            independent_analysis=False, inject_date=False,
            show_math_steps=False, prefer_multiple_sources=False,
            avoid_fabricated_citations=False, json_mode_hint=False,
            safe_inlined_content=False, re2=False, cot_reflection=False,
            readurls=False, compressor=c,
        )
        await c.wait_inflight()
        assert len(seen) == 1
        assert "抚慰套话" in seen[0]
        # 首次请求传输原文
        assert body["messages"][0]["role"] == "system"
        assert "抚慰套话" in body["messages"][0]["content"]

    async def test_no_skills_no_user_system_no_compression(self, tmp_path):
        from deep_proxy.optimization import apply_cheap_optimizations

        c = _make_compressor(tmp_path)

        async def must_not_call(text):
            raise AssertionError("should not call LLM when nothing to compress")

        c._call_llm = must_not_call  # type: ignore[assignment]

        body = {"messages": [{"role": "user", "content": "hi"}]}
        await apply_cheap_optimizations(
            body, avoid_negative_style=False, natural_temperament=False, contextual_register=False,
            assume_good_intent=False, instruction_priority=False,
            independent_analysis=False, reason_genuinely=False,
            inject_date=False, cot_reset=False,
            show_math_steps=False, prefer_multiple_sources=False,
            avoid_fabricated_citations=False, json_mode_hint=False,
            safe_inlined_content=False, re2=False, cot_reflection=False,
            readurls=False, compressor=c,
        )
        # 没有 system 消息插入
        assert all(m.get("role") != "system" for m in body["messages"])

    async def test_multimodal_user_system_skipped(self, tmp_path):
        """用户 system 是 list（多模态）时不压缩，保留原结构。"""
        from deep_proxy.optimization import apply_cheap_optimizations

        c = _make_compressor(tmp_path)
        seen = []

        async def fake_llm(text):
            seen.append(text)
            return "C"

        c._call_llm = fake_llm  # type: ignore[assignment]

        original_sys = [{"type": "text", "text": "multimodal sys"}]
        body = {
            "messages": [
                {"role": "system", "content": original_sys},
                {"role": "user", "content": "hi"},
            ],
        }
        await apply_cheap_optimizations(
            body, avoid_negative_style=True, contextual_register=False,
            assume_good_intent=False, instruction_priority=False,
            independent_analysis=False, inject_date=False,
            show_math_steps=False, prefer_multiple_sources=False,
            avoid_fabricated_citations=False, json_mode_hint=False,
            safe_inlined_content=False, re2=False, cot_reflection=False,
            readurls=False, compressor=c,
        )
        # 多模态 system 原样保留
        sys_msgs = [m for m in body["messages"] if m["role"] == "system"]
        assert any(m["content"] == original_sys for m in sys_msgs)
        # skills 单独插一条新 system 在前
        assert sys_msgs[0]["content"] != original_sys

class TestStripWrapping:
    """合并：覆盖 code-fence、引号、passthrough 三条路径。"""

    def test_strips_wrappers(self):
        from deep_proxy.optimization.compressor import _strip_wrapping
        assert _strip_wrapping("```\nfoo bar\n```") == "foo bar"
        assert _strip_wrapping('"hello"') == "hello"
        assert _strip_wrapping("plain text") == "plain text"
