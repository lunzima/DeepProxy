"""鉴权头识别测试 — OpenAI 风格端点必须同时接受 Bearer 与 x-api-key。

Claude Code 用 ANTHROPIC_API_KEY 配置时只发 x-api-key 头，启动期会探测
/v1/models；如果该端点不认 x-api-key，整条交互都会 401（即使 /v1/messages
路径单独 OK）。
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from deep_proxy import main as main_mod
from deep_proxy.config import DeepSeekConfig, ProxyConfig
from deep_proxy.main import app
from deep_proxy.router import DeepProxyRouter


def _install(api_key: str):
    cfg = ProxyConfig(
        api_key=api_key,
        deepseek=DeepSeekConfig(api_key="sk-upstream", api_base="https://api.deepseek.com"),
    )
    main_mod.config = cfg
    main_mod.router = DeepProxyRouter(cfg)


class TestModelsEndpointAuth:
    def setup_method(self):
        _install("sk-proxy")
        self.client = TestClient(app)

    def test_bearer_accepted(self):
        r = self.client.get("/v1/models", headers={"Authorization": "Bearer sk-proxy"})
        assert r.status_code == 200

    def test_x_api_key_accepted(self):
        r = self.client.get("/v1/models", headers={"x-api-key": "sk-proxy"})
        assert r.status_code == 200

    def test_wrong_bearer_rejected(self):
        r = self.client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_wrong_x_api_key_rejected(self):
        r = self.client.get("/v1/models", headers={"x-api-key": "wrong"})
        assert r.status_code == 401

    def test_no_auth_rejected(self):
        r = self.client.get("/v1/models")
        assert r.status_code == 401


class TestMessagesEndpointAuth:
    """/v1/messages 接受两种风格 — 已经实现，固化为回归测试。"""

    def setup_method(self):
        _install("sk-proxy")
        self.client = TestClient(app)

    def test_x_api_key_accepted(self):
        # 401 不会触发；任何后续上游错误都说明鉴权放行了
        r = self.client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-proxy"},
            json={"model": "claude-3-5-sonnet", "max_tokens": 1, "messages": []},
        )
        assert r.status_code != 401

    def test_bearer_accepted(self):
        r = self.client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer sk-proxy"},
            json={"model": "claude-3-5-sonnet", "max_tokens": 1, "messages": []},
        )
        assert r.status_code != 401

    def test_wrong_key_rejected(self):
        r = self.client.post(
            "/v1/messages",
            headers={"x-api-key": "wrong"},
            json={"model": "claude-3-5-sonnet", "max_tokens": 1, "messages": []},
        )
        assert r.status_code == 401


class TestNoApiKeyConfigured:
    """未配置 api_key 时跳过鉴权（玩具部署/本地测试场景）。"""

    def setup_method(self):
        _install("")
        self.client = TestClient(app)

    def test_models_no_auth_allowed(self):
        r = self.client.get("/v1/models")
        assert r.status_code == 200
