"""验证 OptimizationConfig.strip_client_telemetry 字段默认值与可选关闭。"""
from deep_proxy.config import OptimizationConfig


def test_strip_client_telemetry_default_true():
    cfg = OptimizationConfig()
    assert cfg.strip_client_telemetry is True


def test_strip_client_telemetry_can_be_disabled():
    cfg = OptimizationConfig(strip_client_telemetry=False)
    assert cfg.strip_client_telemetry is False
