"""DeepSeek 错误码映射 — 将 DeepSeek 特有错误转换为 OpenAI 兼容格式。"""

from __future__ import annotations

from typing import Dict

from fastapi import HTTPException
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)


# DeepSeek 官方文档已知的错误码对照
_DEEPSEEK_ERROR_MAP: Dict[int, tuple[int, str]] = {
    400: (400, "invalid_request_error"),
    401: (401, "authentication_error"),
    402: (402, "invalid_request_error"),       # 余额不足
    403: (403, "permission_error"),
    429: (429, "rate_limit_error"),
    500: (500, "api_error"),
    503: (503, "service_unavailable_error"),
}

# DeepSeek 始终不支持的 OpenAI 参数
_UNSUPPORTED_OPENAI_PARAMS = {
    "functions",  # 旧版 OpenAI functions API；DeepSeek 仅支持 tools
    "user",       # DeepSeek 不接受
}


def map_litellm_error(exc: Exception) -> HTTPException:
    """将 LiteLLM 异常映射为标准的 OpenAI 格式 HTTP 错误。"""
    status_code = 500
    error_type = "api_error"
    message = str(exc)

    if isinstance(exc, AuthenticationError):
        status_code, error_type = 401, "authentication_error"
    elif isinstance(exc, RateLimitError):
        status_code, error_type = 429, "rate_limit_error"
    elif isinstance(exc, ServiceUnavailableError):
        status_code, error_type = 503, "service_unavailable_error"
    elif isinstance(exc, APIConnectionError):
        status_code, error_type = 502, "api_connection_error"
        message = "无法连接到 DeepSeek API，请检查网络或 API 地址"
    elif isinstance(exc, APIError):
        status_code = exc.status_code or 500
        error_type = _DEEPSEEK_ERROR_MAP.get(status_code, (500, "api_error"))[1]

    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": status_code,
            }
        },
    )


def strip_unsupported_params(body: dict) -> dict:
    """移除 DeepSeek 不支持的 OpenAI 参数（`functions` / `user`）。

    依据官方文档：V4 全面支持 temperature/top_p/presence_penalty/frequency_penalty
    /response_format/tools/tool_choice/stream_options，无 thinking 模式相关 strip 需求。
    """
    return {k: v for k, v in body.items() if k not in _UNSUPPORTED_OPENAI_PARAMS}


