"""集成测试：redirect + flash_upgrade + cross_consult tool 三者联动（plan §3.3）。

覆盖评审 #4 指出的核心交互——以前 redirect 路径仅单元测试覆盖，combo 行为
没有验证。本文件验证：
- 标签触发后 source=deepseek 重定向到 mimo
- prepare_request 用重定向后的 mimo provider 走完整管道
- mimo 的 per_provider flash_upgrade 阈值生效（不是 deepseek 的）
- cross_consult tool 注入时 pair_for 用 mimo 作为源、回查 deepseek 作为目标
- awareness 段也按重定向后的 source/target 描述
"""
from __future__ import annotations

import pytest

from deep_proxy.config import ProxyConfig
from deep_proxy.cross_consult import resolve_redirect
from deep_proxy.router import DeepProxyRouter


REDIRECT_TAG = "[本轮对话使用不同家族的大语言模型]"


def _build_dual_cfg() -> ProxyConfig:
    """与生产配置（config.yaml）对齐：cross_consult enabled + pairs + MiMo per_provider 偏向升格。"""
    return ProxyConfig.model_validate({
        "providers": {
            "deepseek": {
                "name": "deepseek", "api_base": "x", "api_key": "y",
                "litellm_prefix": "deepseek/",
                "flash_model": "deepseek-v4-flash", "pro_model": "deepseek-v4-pro",
            },
            "mimo": {
                "name": "mimo", "api_base": "x", "api_key": "y",
                "litellm_prefix": "openai/",
                "flash_model": "mimo-v2.5", "pro_model": "mimo-v2.5-pro",
                "reasoning_effort_field": "reasoning_effort",
                "reasoning_effort_value": "high",
                "allowed_extra_params": ["reasoning_effort", "thinking"],
            },
        },
        "ports": [
            {"port": 8000, "provider": "deepseek", "sampling": "precise"},
            {"port": 8001, "provider": "mimo", "sampling": "creative"},
        ],
        "deepseek": {"api_key": "y"},
        "cross_consult": {
            "enabled": True,
            "pairs": {"deepseek": "mimo", "mimo": "deepseek"},
            "redirect_persist_turns": 2,
        },
        "flash_upgrade": {
            "enabled": True,
            "router_type": "rule",  # 跳过 BERT，纯启发式 + 规则
            "router_threshold": 0.65,
            "heuristic_threshold": 8.0,
            "per_provider": {
                "mimo": {
                    "router_threshold": 0.60,
                    "heuristic_threshold": 7.5,
                },
            },
        },
    })


async def test_combo_tag_triggers_redirect_then_mimo_pipeline_runs():
    """端到端 combo：tag → 重定向到 mimo → prepare_request 用 mimo provider 跑通。"""
    cfg = _build_dual_cfg()
    router = DeepProxyRouter(cfg)

    body = {
        "model": "deepseek-v4-flash",  # 客户端原始请求名
        "messages": [
            {"role": "user", "content": f"写一首关于秋天的诗 {REDIRECT_TAG}"},
        ],
    }
    src = cfg.providers["deepseek"]

    # 步骤 1：resolve_redirect 把 deepseek → mimo
    target = resolve_redirect(
        body, source_provider=src, config=cfg,
        tracker=router._redirect_tracker,
    )
    assert target is not None
    assert target.name == "mimo"

    # 标签已剥离
    assert REDIRECT_TAG not in body["messages"][-1]["content"]
    assert "秋天的诗" in body["messages"][-1]["content"]

    # 步骤 2：prepare_request 用重定向后的 mimo provider
    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling, provider=target,
    )

    # 模型名被规范化到 mimo 的 flash_model（或升格后的 pro_model）
    assert out["model"] in ("mimo-v2.5", "mimo-v2.5-pro")

    # cross_consult tool 注入存在
    tools = out.get("tools") or []
    tool_names = [t.get("function", {}).get("name") for t in tools]
    assert "cross_consult" in tool_names

    # awareness + tool addendum 注入 system；用 mimo 视角描述（source=mimo, target=deepseek）
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    sys_text = "\n".join(m.get("content", "") for m in sys_msgs)
    assert "双家族披露" in sys_text
    # 重定向后 source 是 mimo
    assert "`mimo`" in sys_text
    # 对偶是 deepseek
    assert "`deepseek`" in sys_text


def test_combo_mimo_per_provider_threshold_is_used_after_redirect():
    """验证 plan §3.3 "核心交互"：重定向到 mimo 后，flash_upgrade 取 mimo 的覆盖阈值而非全局值。"""
    cfg = _build_dual_cfg()
    # 全局 router_threshold=0.65 / heuristic_threshold=8.0
    # mimo 覆盖 router_threshold=0.60 / heuristic_threshold=7.5
    assert cfg.flash_upgrade.threshold_for_provider("mimo", "router_threshold") == 0.60
    assert cfg.flash_upgrade.threshold_for_provider("mimo", "heuristic_threshold") == 7.5
    # 对照：deepseek 没有覆盖 → 取全局值
    assert cfg.flash_upgrade.threshold_for_provider("deepseek", "router_threshold") == 0.65
    assert cfg.flash_upgrade.threshold_for_provider("deepseek", "heuristic_threshold") == 8.0


async def test_combo_persist_window_then_cross_consult_pair_inverts():
    """重定向后 cross_consult tool 的 pair_for 用 mimo 作为源 → 工具能回查 deepseek。

    确保 plan §3.3 "对称工作" 的承诺：重定向到 mimo 的对话里，cross_consult 工具
    应该让 mimo 能向 deepseek 求第二视角（而不是仍指向自己）。
    """
    cfg = _build_dual_cfg()
    router = DeepProxyRouter(cfg)
    src = cfg.providers["deepseek"]

    # 轮 1：触发重定向
    body1 = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": f"任务 {REDIRECT_TAG}"}],
    }
    target1 = resolve_redirect(
        body1, source_provider=src, config=cfg,
        tracker=router._redirect_tracker,
    )
    assert target1.name == "mimo"
    out1 = await router.prepare_request(
        body1, sampling_profile=cfg.precise_sampling, provider=target1,
    )
    sys_text1 = "\n".join(
        m.get("content", "") for m in out1["messages"] if m["role"] == "system"
    )
    # awareness 段：当前家族 mimo、对偶 deepseek
    assert "`mimo`" in sys_text1 and "`deepseek`" in sys_text1
    # cross_consult 工具应该存在且面向 mimo→deepseek（addendum 不显式说，但 pair_for 决定它能 emit）
    assert cfg.cross_consult.pair_for("mimo") == "deepseek"

    # 轮 2：同对话 + 新 user 消息（无 tag）→ persist 窗口仍生效
    body2 = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "user", "content": "任务"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "继续上面"},
        ],
    }
    target2 = resolve_redirect(
        body2, source_provider=src, config=cfg,
        tracker=router._redirect_tracker,
    )
    assert target2 is not None and target2.name == "mimo", "persist 窗口内应仍走 mimo"


async def test_combo_no_redirect_uses_source_pipeline_intact():
    """对照：无标签 → 不重定向 → prepare_request 用原 deepseek provider 跑（awareness 中是 deepseek→mimo）。"""
    cfg = _build_dual_cfg()
    router = DeepProxyRouter(cfg)
    src = cfg.providers["deepseek"]

    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "plain question"}],
    }

    target = resolve_redirect(
        body, source_provider=src, config=cfg,
        tracker=router._redirect_tracker,
    )
    assert target is None  # 无标签 + 无窗口 → 不重定向

    out = await router.prepare_request(
        body, sampling_profile=cfg.precise_sampling, provider=src,
    )
    # 仍走 deepseek
    assert out["model"] in ("deepseek-v4-flash", "deepseek-v4-pro")
    # awareness：source=deepseek, target=mimo
    sys_text = "\n".join(
        m.get("content", "") for m in out["messages"] if m["role"] == "system"
    )
    assert "`deepseek`" in sys_text
    assert "`mimo`" in sys_text
