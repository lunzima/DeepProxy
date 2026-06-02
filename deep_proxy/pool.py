"""Writing-port 加权模型桶选择器。

逐请求从 PortBinding.model_pool 加权随机选一个 (provider, model)。
无会话粘滞——每次调用独立重掷。选中的 model 保证属于选中的 provider
（由 ProxyConfig.model_validator 在加载期校验），返回后由端点覆盖 body["model"]，
进入既有 prepare_request 管道（flash→可升格，pro→pin）。
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ProxyConfig
    from .providers import PortBinding, Provider


def select_pool_target(
    binding: "PortBinding",
    config: "ProxyConfig",
    *,
    rng: random.Random | random.SystemRandom = random,  # type: ignore[assignment]
) -> tuple["Provider", str]:
    """对 binding.model_pool 加权随机选一个条目，返回 (Provider, model_id)。

    rng 注入以便测试使用 seeded RNG（默认用全局 random 模块）。
    要求 binding.model_pool 非空（调用方在有 pool 时才调本函数）。
    """
    pool = binding.model_pool
    if not pool:
        raise ValueError("select_pool_target 要求 binding.model_pool 非空")
    weights = [e.weight for e in pool]
    entry = rng.choices(pool, weights=weights, k=1)[0]
    provider = config.providers[entry.provider]
    return provider, entry.model


def reconcile_redirected_pool_model(
    selected_model: str | None,
    pre_provider: "Provider | None",
    post_provider: "Provider | None",
) -> str | None:
    """cross-consult redirect 切换 provider 后，对 pool 选中的模型做 tier 保持重映射。

    pool 选中的 (provider, model) 在 redirect 把 provider 翻到异家族后会与新 provider
    不匹配（如选中 deepseek-v4-pro、redirect 到 mimo）。直接交给 prepare_request 的
    normalize 会丢失 tier（pro 被降回 flash）。本函数保持 tier（flash/pro），映射到
    redirect 后 provider 的对应档，从而维持"pro 起始 → pin 在 pro"不变式。

    无 pool（selected_model None）或 provider 未切换（post is pre）时原样返回。
    """
    if selected_model is None or post_provider is pre_provider:
        return selected_model
    if pre_provider is None or post_provider is None:
        return selected_model
    is_pro = selected_model == pre_provider.pro_model
    return post_provider.pro_model if is_pro else post_provider.flash_model
