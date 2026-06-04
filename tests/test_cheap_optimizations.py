"""测试 in-process 廉价提示词优化（CoT Reflection / RE2 / readurls）—— 忠实于 optillm。"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from deep_proxy.optimization import (
    apply_cheap_optimizations,
    extract_cot_output,
)
from deep_proxy.optimization.skills_general import _COT_SYSTEM_PROMPT


# OptimizationConfig 的 18 个 skill flag 字段名（与 deep_proxy.config 同步）
_OPT_FLAG_FIELDS = (
    "avoid_negative_style", "natural_temperament", "contextual_register",
    "assume_good_intent", "instruction_priority", "independent_analysis",
    "reason_genuinely", "inject_date", "cot_reset",
    "show_math_steps", "prefer_multiple_sources", "avoid_fabricated_citations",
    "json_mode_hint", "safe_inlined_content",
    "re2", "cot_reflection", "readurls",
    "tool_call_chinese_cot",
)


def _opt(**overrides):
    """构造 duck-typed opt 对象（全 flag 默认 False，按需 overrides 启用）。

    apply_cheap_optimizations 通过 opt.<field> 读取，SimpleNamespace 完美匹配。
    """
    defaults = {name: False for name in _OPT_FLAG_FIELDS}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_body(messages, **extra):
    return {"messages": messages, **extra}


async def _apply(body, **kwargs):
    """缩写：默认全关，按需启用单项。"""
    return await apply_cheap_optimizations(body, opt=_opt(**kwargs))


class TestRe2:
    async def test_duplicates_last_user_message(self):
        b = _make_body([{"role": "user", "content": "What is 1+1?"}])
        await _apply(b, re2=True)
        assert b["messages"][0]["content"] == (
            "What is 1+1?\n请再阅读一遍上面的内容，然后作答：\nWhat is 1+1?"
        )

    async def test_only_last_user_modified(self):
        b = _make_body([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ])
        await _apply(b, re2=True)
        assert b["messages"][0]["content"] == "first"
        assert b["messages"][2]["content"] == (
            "second\n请再阅读一遍上面的内容，然后作答：\nsecond"
        )

    async def test_idempotent(self):
        original = "q\n请再阅读一遍上面的内容，然后作答：\nq"
        b = _make_body([{"role": "user", "content": original}])
        await _apply(b, re2=True)
        assert b["messages"][0]["content"] == original

    async def test_skips_non_string_content(self):
        b = _make_body([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        await _apply(b, re2=True)
        assert b["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


class TestCotReflectionEligibility:
    async def test_applied_when_thinking_disabled_and_non_stream(self):
        b = _make_body(
            [{"role": "user", "content": "x"}],
            thinking={"type": "disabled"},
        )
        await _apply(b, cot_reflection=True)
        assert b["messages"][0]["role"] == "system"
        assert "<output>" in b["messages"][0]["content"]
        assert b["_deepproxy_strip_cot"] is True

    async def test_skipped_when_thinking_not_disabled(self):
        """enabled 或 missing 都不应触发 cot_reflection。"""
        for body_extra in ({"thinking": {"type": "enabled"}}, {}):
            b = _make_body([{"role": "user", "content": "x"}], **body_extra)
            await _apply(b, cot_reflection=True)
            assert b["messages"][0]["role"] == "user"
            assert "_deepproxy_strip_cot" not in b

    async def test_skipped_when_streaming(self):
        b = _make_body(
            [{"role": "user", "content": "x"}],
            thinking={"type": "disabled"},
            stream=True,
        )
        await _apply(b, cot_reflection=True)
        assert b["messages"][0]["role"] == "user"
        assert "_deepproxy_strip_cot" not in b

    async def test_merges_into_existing_system(self):
        b = _make_body(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "x"},
            ],
            thinking={"type": "disabled"},
        )
        await _apply(b, cot_reflection=True)
        sys_content = b["messages"][0]["content"]
        assert "<output>" in sys_content
        assert "Be concise." in sys_content


class TestSkipConditions:
    async def test_skip_when_tool_invocation_present(self):
        """tools 或 tool_choice 任一出现都跳过整条 D 类管道。"""
        for extra in ({"tools": [{"type": "function", "function": {"name": "x"}}],
                       "thinking": {"type": "disabled"}},
                      {"tool_choice": "auto"}):
            b = _make_body([{"role": "user", "content": "hi"}], **extra)
            await _apply(b, cot_reflection=True, re2=True, readurls=True)
            assert b["messages"][0]["content"] == "hi"
            assert "_deepproxy_strip_cot" not in b

    async def test_skip_when_no_messages(self):
        b = {"messages": []}
        await _apply(b, cot_reflection=True, re2=True, readurls=True)
        assert b["messages"] == []


class TestExtractCotOutput:
    def test_extracts_output_tag(self):
        text = (
            "<thinking>step 1, step 2</thinking>\n"
            "<output>The answer is 42.</output>"
        )
        assert extract_cot_output(text) == "The answer is 42."

    def test_open_output_tag_only(self):
        text = "<thinking>...</thinking>\n<output>partial answer"
        assert extract_cot_output(text) == "partial answer"

    def test_passthrough_when_no_or_empty_tags(self):
        assert extract_cot_output("Direct answer without tags.") == "Direct answer without tags."
        assert extract_cot_output("") == ""

    def test_empty_output_strips_tags(self):
        text = "<thinking>x</thinking><output></output>"
        assert extract_cot_output(text) == ""
        text2 = "<thinking>step 1</thinking><reflection>oops</reflection><output>  </output>"
        assert extract_cot_output(text2) == ""


class TestSkillJsonMode:
    async def test_injects_when_json_object_response_format(self):
        b = _make_body(
            [{"role": "user", "content": "give me data"}],
            response_format={"type": "json_object"},
        )
        await _apply(b, json_mode_hint=True)
        assert b["messages"][0]["role"] == "system"

    async def test_no_op_when_not_json_object(self):
        b = _make_body(
            [{"role": "user", "content": "give me data"}],
            response_format={"type": "text"},
        )
        await _apply(b, json_mode_hint=True)
        assert b["messages"][0]["role"] == "user"

    async def test_no_op_when_no_response_format(self):
        b = _make_body([{"role": "user", "content": "x"}])
        await _apply(b, json_mode_hint=True)
        assert b["messages"][0]["role"] == "user"


class TestSkillDate:
    async def test_injects_today(self):
        b = _make_body([{"role": "user", "content": "x"}])
        await _apply(b, inject_date=True)
        sys = b["messages"][0]
        assert sys["role"] == "system"
        # 用正则验证日期格式 YYYY-MM-DD
        import re
        assert re.search(r"今天是 \d{4}-\d{2}-\d{2}", sys["content"])

    async def test_idempotent(self):
        b = _make_body([{"role": "user", "content": "x"}])
        await _apply(b, inject_date=True)
        first = b["messages"][0]["content"]
        await _apply(b, inject_date=True)
        # 第二次不应再嵌套一层
        assert b["messages"][0]["content"] == first

    async def test_merges_into_existing_system(self):
        b = _make_body([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "x"},
        ])
        await _apply(b, inject_date=True)
        sys_content = b["messages"][0]["content"]
        assert "今天是" in sys_content
        assert "Be concise." in sys_content


class TestSkillSafeInlinedContent:
    async def test_injects_when_inlined_content_present(self):
        b = _make_body([
            {"role": "user", "content": "Check https://x.com [Content from x.com: foo]"},
        ])
        await _apply(b, safe_inlined_content=True)
        assert b["messages"][0]["role"] == "system"

    async def test_no_op_without_inlined_content(self):
        b = _make_body([{"role": "user", "content": "no links"}])
        await _apply(b, safe_inlined_content=True)
        assert b["messages"][0]["role"] == "user"


class TestSkillAntiOverRefusal:
    """借鉴自 grok-prompts 的 3 个 skills —— 创作场景实际是积极改善。"""

    async def test_each_skill_injects_system(self):
        for flag in ("avoid_negative_style", "assume_good_intent",
                     "instruction_priority"):
            b = _make_body([{"role": "user", "content": "x"}])
            await _apply(b, **{flag: True})
            sys_msgs = [m for m in b["messages"] if m["role"] == "system"]
            assert sys_msgs and sys_msgs[0]["content"], f"{flag} 未注入"

    async def test_three_skills_merge_into_single_system(self):
        """合并语义：3 个一起开仍然是单条 system。"""
        b = _make_body([{"role": "user", "content": "x"}])
        await _apply(b, avoid_negative_style=True,
                     assume_good_intent=True, instruction_priority=True)
        sys_msgs = [m for m in b["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1

class TestSkillVerification:
    """求证 / 反幻觉 skills 启用后会注入 system 消息。"""

    async def test_each_skill_injects_system(self):
        for flag in ("show_math_steps", "avoid_fabricated_citations",
                     "independent_analysis", "prefer_multiple_sources",
                     "reason_genuinely", "cot_reset"):
            b = _make_body([{"role": "user", "content": "x"}])
            await _apply(b, **{flag: True})
            sys_msgs = [m for m in b["messages"] if m["role"] == "system"]
            assert sys_msgs and sys_msgs[0]["content"], f"{flag} 未注入"


class TestReadUrls:
    """readurls 用 mock httpx client，验证抓取与内联逻辑（不依赖外网）。"""

    async def test_inlines_fetched_html_content(self):
        html = b"<html><body><h1>Title</h1><p>Body para.</p></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html, headers={"content-type": "text/html"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b = _make_body([{"role": "user", "content": "Look at https://example.com/page"}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        content = b["messages"][0]["content"]
        assert "[Content from example.com:" in content

    async def test_failed_fetch_is_silent(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            original = "Try https://broken.test/x please"
            b = _make_body([{"role": "user", "content": original}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        # fail-open：失败不修改
        assert b["messages"][0]["content"] == original

    async def test_no_url_no_op(self):
        b = _make_body([{"role": "user", "content": "no link here"}])
        await _apply(b, readurls=True)
        assert b["messages"][0]["content"] == "no link here"

    async def test_idempotent(self):
        already = "Visit https://x.com [Content from x.com: cached]"
        b = _make_body([{"role": "user", "content": already}])
        await _apply(b, readurls=True)
        assert b["messages"][0]["content"] == already

    async def test_multiple_urls_all_inlined(self):
        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            html = f"<p>page on {host}</p>".encode()
            return httpx.Response(200, content=html, headers={"content-type": "text/html"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b = _make_body([
                {"role": "user", "content": "Compare https://a.com and https://b.com"},
            ])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        c = b["messages"][0]["content"]
        assert "[Content from a.com:" in c
        assert "[Content from b.com:" in c

    async def test_truncates_long_content(self):
        long_html = b"<p>" + b"X" * 50000 + b"</p>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=long_html, headers={"content-type": "text/html"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b = _make_body([{"role": "user", "content": "https://big.test/x"}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        c = b["messages"][0]["content"]
        # 截断后应包含 ... 后缀
        assert "..." in c
        # 不应超过预期上限 + 一些标签字符
        assert len(c) < 12000

    async def test_non_text_content_type_skipped(self):
        # PDF / 图像等二进制 Content-Type 应被跳过，不当 HTML 解析
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n binary data",
                headers={"content-type": "application/pdf"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            original = "See https://files.test/spec.pdf"
            b = _make_body([{"role": "user", "content": original}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        # 内容未被注入（不应有 [Content from ...]）
        assert "[Content from" not in b["messages"][0]["content"]

    async def test_max_urls_per_message_capped(self):
        # 超过 _READURLS_MAX_PER_MSG（=6）时仅前若干个被抓取
        from deep_proxy.optimization.skills_transform import _READURLS_MAX_PER_MSG

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            return httpx.Response(200, content=f"<p>{host}</p>".encode(),
                                  headers={"content-type": "text/html"})

        urls = [f"https://h{i}.test/" for i in range(_READURLS_MAX_PER_MSG + 4)]
        text = " ".join(urls)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b = _make_body([{"role": "user", "content": text}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        c = b["messages"][0]["content"]
        # 至多 _READURLS_MAX_PER_MSG 个 [Content from ...]
        assert c.count("[Content from") == _READURLS_MAX_PER_MSG

    async def test_one_url_failure_does_not_block_others(self):
        # 一个 URL 抛网络异常，另一个正常 — 后者仍应被内联
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "fail.test":
                raise httpx.ConnectError("simulated")
            return httpx.Response(200, content=b"<p>ok</p>",
                                  headers={"content-type": "text/html"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b = _make_body([{"role": "user",
                             "content": "A: https://fail.test/x B: https://ok.test/y"}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        c = b["messages"][0]["content"]
        # 失败 URL 未注入；成功 URL 已注入
        assert "[Content from ok.test:" in c
        assert "[Content from fail.test:" not in c

    async def test_oversized_response_truncated_not_oom(self):
        # 远超 _READURLS_MAX_BYTES 的响应应被截断而非读完
        big = b"<p>" + (b"X" * (5 * 1024 * 1024)) + b"</p>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big, headers={"content-type": "text/html"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b = _make_body([{"role": "user", "content": "https://huge.test/x"}])
            await apply_cheap_optimizations(b, opt=_opt(readurls=True), http_client=client)
        c = b["messages"][0]["content"]
        # 注入成功（截断后仍是有效内容），且包含 ... 截断标记
        assert "[Content from huge.test:" in c
        assert "..." in c


class TestReadUrlsHelpers:
    """A7 重构后新增的纯函数 helper 单元测试（无 I/O，快）。"""

    def test_extract_urls_dedup_and_trailing_punct_strip(self):
        from deep_proxy.optimization.skills_transform import _extract_urls
        content = "see https://a.com, and https://a.com. also https://b.com!"
        urls = _extract_urls(content)
        # 去重 + 尾部标点剥离
        assert urls == ["https://a.com", "https://b.com"]

    def test_extract_urls_respects_cap(self):
        from deep_proxy.optimization.skills_transform import (
            _extract_urls, _READURLS_MAX_PER_MSG,
        )
        urls = " ".join(f"https://h{i}.test/" for i in range(_READURLS_MAX_PER_MSG + 4))
        result = _extract_urls(urls)
        assert len(result) == _READURLS_MAX_PER_MSG

    def test_extract_urls_no_urls_returns_empty(self):
        from deep_proxy.optimization.skills_transform import _extract_urls
        assert _extract_urls("no link here") == []

    def test_substitute_urls_inlines_successful_results(self):
        from deep_proxy.optimization.skills_transform import _substitute_urls
        urls = ["https://a.com", "https://b.com"]
        results = ["page A", "page B"]
        out = _substitute_urls("see https://a.com and https://b.com", urls, results)
        assert "[Content from a.com: page A]" in out
        assert "[Content from b.com: page B]" in out

    def test_substitute_urls_skips_exceptions_and_empty(self):
        from deep_proxy.optimization.skills_transform import _substitute_urls
        urls = ["https://a.com", "https://b.com", "https://c.com"]
        results = [Exception("network"), "", "real content"]
        out = _substitute_urls(
            "x https://a.com y https://b.com z https://c.com", urls, results,
        )
        # a.com 异常 / b.com 空 都不内联；c.com 正常内联
        assert "[Content from a.com" not in out
        assert "[Content from b.com" not in out
        assert "[Content from c.com: real content]" in out

    def test_substitute_urls_reraises_cancelled_error(self):
        import asyncio
        from deep_proxy.optimization.skills_transform import _substitute_urls
        urls = ["https://a.com"]
        results = [asyncio.CancelledError("cancelled")]
        with pytest.raises(asyncio.CancelledError):
            _substitute_urls("https://a.com", urls, results)

    def test_substitute_urls_prefix_url_not_corrupted(self):
        """一个 URL 是另一个的前缀时，短 URL 的替换不得切进长 URL 中间。"""
        from deep_proxy.optimization.skills_transform import _substitute_urls
        urls = ["https://a.com", "https://a.com/page"]
        results = ["SHORT", "LONG"]
        out = _substitute_urls("see https://a.com/page here", urls, results)
        # 长 URL 完整保留并正确内联；短 URL（无独立出现）不破坏它
        assert "https://a.com/page [Content from a.com: LONG]" in out
        assert "[SHORT]/page" not in out          # 旧 bug：短替换切进长 URL

    def test_substitute_urls_snippet_with_backslash_safe(self):
        """抓取片段含正则替换特殊串（\\1 等）不应破坏替换。"""
        from deep_proxy.optimization.skills_transform import _substitute_urls
        out = _substitute_urls("x https://a.com y", ["https://a.com"], [r"\1\g<0>"])
        assert r"[Content from a.com: \1\g<0>]" in out


class TestCreativeMode:
    """验证 mode="creative" 时使用新 skills 路径。

    - A 组：_SKILL_AVOID_AI_TICS 替代 _SKILL_AVOID_NEGATIVE_STYLE_CODING
    - B 组：求证/反幻觉 skills 全部跳过
    """

    def _find_system_text(self, body):
        for msg in body.get("messages", []):
            if msg.get("role") == "system":
                return msg.get("content", "")
        return ""

    async def test_creative_mode_uses_ai_tics_not_coding(self):
        """creative mode 下 avoid_negative_style 应注入通用 _SKILL_AVOID_AI_TICS，
        不含旧 coding 版的特定指令。"""
        b = _make_body([{"role": "user", "content": "hello"}])
        await apply_cheap_optimizations(b, opt=_opt(avoid_negative_style=True), mode="creative")
        system_text = self._find_system_text(b)
        # 通用版含"无实质确认"（从旧 coding 版合并进来的）
        assert "无实质确认" in system_text
        # 通用版含旧创作版的特性
        assert "说教套话" in system_text
        # 通用版不含旧 coding 版的特定情感删除指令
        assert "把情感支持转成推进事情的方式" not in system_text
        assert "把总结感受的空间留给读者" not in system_text

    async def test_coding_mode_uses_same_ai_tics(self):
        """coding mode 使用与 creative 相同的 _SKILL_AVOID_AI_TICS 通用版，
        不再使用旧的编码版。"""
        b = _make_body([{"role": "user", "content": "hello"}])
        await apply_cheap_optimizations(b, opt=_opt(avoid_negative_style=True), mode="coding")
        system_text = self._find_system_text(b)
        # 通用版有的（coding 也应有）
        assert "无实质确认" in system_text
        assert "说教套话" in system_text
        assert "情感抚慰套话" in system_text
        # 旧 coding 版的特定指令已迁移到通用版
        assert "把情感支持转成推进事情的方式" not in system_text

    async def test_creative_mode_uses_narrator_stance(self):
        """creative mode 下 natural_temperament 应注入 _SKILL_NARRATOR_STANCE。"""
        b = _make_body([{"role": "user", "content": "hello"}])
        await apply_cheap_optimizations(b, opt=_opt(natural_temperament=True), mode="creative")
        system_text = self._find_system_text(b)
        assert "叙事者不替他们降温" in system_text
        assert "以动词和名词为主力" in system_text
        # 不含旧 personality 的"与人交流时保留一定的分寸"
        assert "与人交流时保留一定的分寸" not in system_text

    async def test_creative_mode_skips_b_group(self):
        """creative mode 下 B 组求证 skills 全部跳过。"""
        b = _make_body([{"role": "user", "content": "hello"}])
        await apply_cheap_optimizations(
            b, opt=_opt(show_math_steps=True, prefer_multiple_sources=True,
                        avoid_fabricated_citations=True),
            mode="creative",
        )
        system_text = self._find_system_text(b)
        # B 组全文不应出现
        assert "推导过程和关键中间步骤" not in system_text  # show_math_steps
        assert "交叉验证与权衡" not in system_text  # prefer_multiple_sources
        assert "逐字引文" not in system_text  # avoid_fabricated_citations

    async def test_coding_mode_includes_b_group(self):
        """mode="coding" 下 B 组求证 skills 正常注入。"""
        b = _make_body([{"role": "user", "content": "hello"}])
        await apply_cheap_optimizations(
            b, opt=_opt(show_math_steps=True, prefer_multiple_sources=True,
                        avoid_fabricated_citations=True),
            mode="coding",
        )
        system_text = self._find_system_text(b)
        assert "推导过程和关键中间步骤" in system_text
        assert "交叉验证与权衡" in system_text
        assert "逐字引文" in system_text
