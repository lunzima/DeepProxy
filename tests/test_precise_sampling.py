"""测试 PreciseSamplingConfig 高确定性采样预设。

Unsloth-validated 区间，强确定性 + 微抖动，专用于：
- 编程 / 数学 / 逻辑推理（未来若加请求时切换机制可用）
- 提示词压缩器（已经在 router 启动时注入）
"""
from __future__ import annotations

import pytest

from deep_proxy.config import (
    PreciseSamplingConfig,
    DeepSeekConfig,
    OptimizationConfig,
    ProxyConfig,
)


class TestRouterInstantiatesCompressorWithProfile:
    """router 启动时应把 precise_sampling 注入给压缩器。"""

    async def test_compressor_receives_precise_sampling_sampling(self):
        from deep_proxy.router import DeepProxyRouter

        cfg = ProxyConfig(
            deepseek=DeepSeekConfig(api_key="sk"),
            optimization=OptimizationConfig(enabled=True, compress_skills=True),
        )
        r = DeepProxyRouter(cfg)
        assert r._compressor is not None
        # 注入的就是 ProxyConfig.precise_sampling
        assert r._compressor._sampling is cfg.precise_sampling


class TestSampleInRange:
    """共享 helper：仅测退化分支（in-range 已被 creative_sampling 集成测覆盖）。"""

    def test_degenerate_returns_lo(self):
        from deep_proxy.utils import sample_in_range
        assert sample_in_range(0.95, 0.95) == 0.95
        # hi < lo 也退化为 lo（防御）
        assert sample_in_range(0.7, 0.3) == 0.7
