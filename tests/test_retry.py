"""测试通用重试退避。"""
from __future__ import annotations

import pytest

from deep_proxy.utils import retry_async as _retry_async
from deep_proxy.litellm_client import _is_retryable_litellm
from litellm.exceptions import RateLimitError, APIError


class TestRetryAsync:
    async def test_succeeds_first_try(self):
        async def ok():
            return 42

        result = await _retry_async(
            ok, max_retries=3, backoff_base=0.0, is_retryable=lambda e: True, label="t",
        )
        assert result == 42

    async def test_retries_then_succeeds(self):
        attempts = {"n": 0}

        async def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("transient")
            return "done"

        result = await _retry_async(
            flaky, max_retries=5, backoff_base=0.0, is_retryable=lambda e: True, label="t",
        )
        assert result == "done"
        assert attempts["n"] == 3

    async def test_gives_up_after_max(self):
        attempts = {"n": 0}

        async def fail():
            attempts["n"] += 1
            raise ValueError("nope")

        with pytest.raises(ValueError):
            await _retry_async(
                fail, max_retries=2, backoff_base=0.0, is_retryable=lambda e: True, label="t",
            )
        assert attempts["n"] == 3  # initial + 2 retries

    async def test_non_retryable_immediately_raises(self):
        attempts = {"n": 0}

        async def fail():
            attempts["n"] += 1
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError):
            await _retry_async(
                fail, max_retries=5, backoff_base=0.0,
                is_retryable=lambda e: not isinstance(e, RuntimeError),
                label="t",
            )
        assert attempts["n"] == 1


class TestIsRetryableLitellm:
    def test_rate_limit(self):
        assert _is_retryable_litellm(RateLimitError(message="x", llm_provider="deepseek", model="x")) is True

    def test_api_error_500(self):
        e = APIError(status_code=500, message="x", llm_provider="deepseek", model="x")
        assert _is_retryable_litellm(e) is True

    def test_api_error_400_not_retryable(self):
        e = APIError(status_code=400, message="x", llm_provider="deepseek", model="x")
        assert _is_retryable_litellm(e) is False

    def test_other_exception(self):
        assert _is_retryable_litellm(ValueError("x")) is False
