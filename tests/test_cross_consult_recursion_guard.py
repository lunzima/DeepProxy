"""Cross-Consult 递归防护：consult 内部调用不重复注入工具。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.cross_consult.config import CrossConsultConfig
from deep_proxy.cross_consult.executor import execute_consult
from deep_proxy.providers import Provider


@pytest.fixture
def cfg_full():
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {"name": "deepseek", "api_base": "x", "api_key": "y",
                         "litellm_prefix": "deepseek/", "flash_model": "deepseek-v4-flash",
                         "pro_model": "deepseek-v4-pro"},
            "mimo": {"name": "mimo", "api_base": "x", "api_key": "y",
                     "litellm_prefix": "openai/", "flash_model": "mimo-v2.5",
                     "pro_model": "mimo-v2.5-pro"},
        },
        "ports": [{"port": 8000, "provider": "deepseek", "sampling": "precise"}],
        "deepseek": {"api_key": "y"},
        "cross_consult": {"enabled": True, "pairs": {"deepseek": "mimo", "mimo": "deepseek"}},
    })


async def test_executor_body_has_recursion_sentinel(cfg_full):
    """executor 发出的 body 必须含 _deepproxy_cross_consult_internal=True。"""
    target = cfg_full.providers["mimo"]
    cc = cfg_full.cross_consult

    captured = {}

    async def fake_iter(config, body, *, provider=None):
        captured["body"] = body
        yield {"choices": [{"index": 0,
                            "delta": {"content": "ok"},
                            "finish_reason": "stop"}]}

    with patch("deep_proxy.cross_consult.streaming.iter_litellm_chunks",
               new=fake_iter):
        await execute_consult(
            question="hi", context=None,
            target_provider=target, config=cfg_full, cc_config=cc,
        )

    assert captured["body"].get("_deepproxy_cross_consult_internal") is True


async def test_prepare_request_with_sentinel_does_not_inject(cfg_full):
    """带 sentinel 的 body 进入 prepare_request 时，注入被跳过。"""
    from deep_proxy.router import DeepProxyRouter
    router = DeepProxyRouter(cfg_full)

    body = {
        "model": "mimo-v2.5-pro",
        "messages": [
            {"role": "system", "content": "consultant"},
            {"role": "user", "content": "what is X?"},
        ],
        "_deepproxy_cross_consult_internal": True,
    }
    out = await router.prepare_request(
        body, sampling_profile=cfg_full.creative_sampling,
        provider=cfg_full.providers["mimo"],
    )
    tools = out.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    assert "cross_consult" not in names
    # system prompt 也不应被追加增量
    sys_text = " ".join(
        m.get("content", "") for m in out["messages"] if m["role"] == "system"
    )
    # 既不应注入"[DeepProxy]"前缀的工具说明，也不应包含工具名提示
    assert "[DeepProxy]" not in sys_text
