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


class TestRouterInstantiatesCompressor:
    """router 启动时（optimization + compress_skills 开启）应构造压缩器。

    压缩器不再接收 sampling profile——压缩调用用固定确定性参数（temperature=0.1），
    见 compressor._build_compress_kwargs；故此处只断言压缩器被正确构造并用配置的
    compressor_model。
    """

    async def test_compressor_constructed_with_configured_model(self):
        from deep_proxy.router import DeepProxyRouter

        cfg = ProxyConfig(
            deepseek=DeepSeekConfig(api_key="sk"),
            optimization=OptimizationConfig(enabled=True, compress_skills=True),
        )
        r = DeepProxyRouter(cfg)
        assert r._compressor is not None
        assert r._compressor._model == cfg.optimization.compressor_model


class TestSampleInRange:
    """共享 helper：仅测退化分支（in-range 已被 creative_sampling 集成测覆盖）。"""

    def test_degenerate_returns_lo(self):
        from deep_proxy.utils import sample_in_range
        assert sample_in_range(0.95, 0.95) == 0.95
        # hi < lo 也退化为 lo（防御）
        assert sample_in_range(0.7, 0.3) == 0.7
