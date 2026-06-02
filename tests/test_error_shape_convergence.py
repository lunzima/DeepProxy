"""端点错误体形状趋同：两个端点都返回**协议干净**的错误体（无 FastAPI 默认的
`{"detail": ...}` 包裹），各自符合自家协议规范。

- OpenAI `/v1/chat/completions`：顶层 `{"error": {message,type,param,code}}`
- Anthropic `/v1/messages`：顶层 `{"type":"error","error":{type,message}}`

二者唯一差异是协议规范要求的形状（排除项），框架包裹/泄漏对方形状都属于应消除的
非协议分歧。
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.testclient import TestClient

from deep_proxy import main as main_mod
from deep_proxy.config import DeepSeekConfig, ProxyConfig
from deep_proxy.main import app
from deep_proxy.router import DeepProxyRouter


def _install_router():
    cfg = ProxyConfig(
        api_key=None,  # 关闭代理鉴权，专注错误体形状
        deepseek=DeepSeekConfig(api_key="sk-upstream", api_base="https://api.deepseek.com"),
    )
    main_mod.config = cfg
    main_mod.router = DeepProxyRouter(cfg)
    return main_mod


def _raise_429(*a, **k):
    # 模拟 map_litellm_error 产出的 OpenAI 形状 HTTPException（两个端点上游错误同源）
    raise HTTPException(
        status_code=429,
        detail={"error": {"message": "rate limited", "type": "rate_limit_error",
                          "param": None, "code": 429}},
    )


class TestOpenAIEndpointErrorShape:
    def setup_method(self):
        m = _install_router()
        self.client = TestClient(app, raise_server_exceptions=False)

        async def _r(*a, **k):
            _raise_429()
        m.router.chat_completions = _r

    def test_upstream_error_clean_openai_body(self):
        r = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 429
        body = r.json()
        # 顶层 error（非 {"detail": ...} 包裹）
        assert "detail" not in body
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["message"] == "rate limited"
        assert body["error"]["code"] == 429


class TestAnthropicEndpointErrorShape:
    def setup_method(self):
        m = _install_router()
        self.client = TestClient(app, raise_server_exceptions=False)

        async def _r(*a, **k):
            _raise_429()
        m.router.chat_completions = _r

    def test_upstream_error_clean_anthropic_body(self):
        r = self.client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 429
        body = r.json()
        assert "detail" not in body
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["message"] == "rate limited"
        # 不泄漏 OpenAI 专属字段
        assert "param" not in body["error"]
        assert "code" not in body["error"]


class TestAnthropicEndpointAllErrorPathsConverge:
    """全局 handler 应让 /v1/messages 的**所有** raise 点（prepare_request / 503 未就绪
    等，不止上游 chat_completions）都返回 Anthropic 形状——否则 prepare 阶段抛错会
    经全局 handler 泄漏 OpenAI 形状给 Anthropic 客户端。"""

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(app, raise_server_exceptions=False)

    def test_prepare_request_error_returns_anthropic_shape(self):
        m = _install_router()
        client = self._client()

        async def _raise_prepare(*a, **k):
            _raise_429()

        m.router.prepare_request = _raise_prepare
        r = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 429
        body = r.json()
        assert "detail" not in body
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert "code" not in body["error"]

    def test_internal_500_returns_anthropic_shape(self):
        """非 HTTPException 的内部异常（except Exception → _internal_error_openai 产 OpenAI
        形状 500）经全局 handler 按 /v1/messages 收敛为 Anthropic 形状。"""
        m = _install_router()
        client = self._client()

        async def _boom(*a, **k):
            raise RuntimeError("kaboom")

        m.router.chat_completions = _boom
        r = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 500
        body = r.json()
        assert "detail" not in body
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"
        assert "kaboom" in body["error"]["message"]
        assert "code" not in body["error"]

    def test_not_ready_503_returns_anthropic_shape(self):
        _install_router()
        from deep_proxy import main as main_mod
        main_mod.router = None  # 触发 _ensure_router_ready 的 503（字符串 detail）
        client = self._client()
        try:
            r = client.post(
                "/v1/messages",
                json={"model": "claude-x", "max_tokens": 16,
                      "messages": [{"role": "user", "content": "hi"}]},
            )
        finally:
            _install_router()  # 复原全局状态，避免污染其它测试
        assert r.status_code == 503
        body = r.json()
        # Anthropic 端点不应返回"两不像"的 {"detail": "..."}；应是 Anthropic 形状
        assert "detail" not in body
        assert body["type"] == "error"
        assert isinstance(body["error"]["message"], str) and body["error"]["message"]
