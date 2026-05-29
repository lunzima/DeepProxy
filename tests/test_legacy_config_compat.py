"""老 config.yaml 在新代码下的兼容性测试。"""
from __future__ import annotations

import pytest
import yaml

from deep_proxy.config import ProxyConfig, normalize_legacy_config


def test_legacy_yaml_loads_with_normalize(tmp_path):
    """老格式 yaml（仅 deepseek + coding_port + writing_port）能加载并自动 normalize。"""
    yaml_text = """\
coding_port: 8000
writing_port: 8001
deepseek:
  api_key: sk-legacy
  api_base: https://api.deepseek.com
optimization:
  enabled: true
"""
    f = tmp_path / "config.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    cfg = ProxyConfig.from_yaml(f)
    assert "deepseek" in cfg.providers
    assert cfg.providers["deepseek"].api_key == "sk-legacy"
    assert len(cfg.ports) == 2
    assert cfg.provider_for_port(8000).name == "deepseek"
    assert cfg.provider_for_port(8001).name == "deepseek"


def test_compressor_credentials_resolve_from_providers_block(tmp_path):
    """新格式 yaml 仅在 providers.deepseek 配置 api_key，无顶层 deepseek 块。

    Router._compressor 必须从 providers 派生凭据，不能因顶层 deepseek 字段
    为空就误触"compress_skills 但未配置 api_key"警告 + 禁用压缩器。
    回归 commit aa90abe 后用户上报的虚假警告。
    """
    import logging
    from deep_proxy.router import DeepProxyRouter

    yaml_text = """\
providers:
  deepseek:
    name: deepseek
    api_base: https://api.deepseek.com
    api_key: sk-providers-block
    litellm_prefix: deepseek/
    flash_model: deepseek-v4-flash
    pro_model: deepseek-v4-pro
ports:
  - port: 8000
    provider: deepseek
    sampling: precise
optimization:
  enabled: true
  compress_skills: true
flash_upgrade:
  enabled: false
"""
    f = tmp_path / "config.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    cfg = ProxyConfig.from_yaml(f)
    # 顶层 deepseek 块缺失 → 默认为空（legacy fallback path）
    assert cfg.deepseek.api_key == ""
    # 但 providers.deepseek 有 key
    assert cfg.providers["deepseek"].api_key == "sk-providers-block"

    # 关键：router 构造 compressor 应从 providers 解析，不触发警告
    records = []
    class Catcher(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    h = Catcher()
    logging.getLogger("deep_proxy.router").addHandler(h)
    try:
        router = DeepProxyRouter(cfg)
    finally:
        logging.getLogger("deep_proxy.router").removeHandler(h)

    assert router._compressor is not None, (
        "compressor 应正常构造（providers.deepseek.api_key 已提供）"
    )
    # 不应有"未配置 DeepSeek api_key"的虚假警告
    assert not any(
        "未配置 DeepSeek api_key" in r for r in records
    ), f"不应触发虚假警告；实际 records: {records}"


def test_new_yaml_loads_without_double_normalize(tmp_path):
    """新格式 yaml 不被 normalize 干扰。"""
    yaml_text = """\
providers:
  deepseek:
    name: deepseek
    api_base: https://api.deepseek.com
    api_key: sk-new
    litellm_prefix: deepseek/
    flash_model: deepseek-v4-flash
    pro_model: deepseek-v4-pro
  mimo:
    name: mimo
    api_base: https://token-plan-cn.xiaomimimo.com/v1
    api_key: tp-new
    litellm_prefix: openai/
    flash_model: mimo-v2.5
    pro_model: mimo-v2.5-pro
    reasoning_effort_field: reasoning_effort
    reasoning_effort_value: high
    max_output_tokens: 128000
ports:
  - port: 8000
    provider: deepseek
    sampling: precise
  - port: 8001
    provider: mimo
    sampling: creative
deepseek:
  api_key: sk-new
"""
    f = tmp_path / "config.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    cfg = ProxyConfig.from_yaml(f)
    assert cfg.provider_for_port(8001).name == "mimo"
    assert cfg.providers["mimo"].api_key == "tp-new"
