"""DeepProxy 服务器启动入口（双端口绑定）。

- coding_port (默认 8000) → precise_sampling profile
- writing_port (默认 8001) → creative_sampling profile
  写作篮在 dynamic_baskets 层按 optimization.writing_basket_kind
  （creative / general）切换；采样参数与端口数量无关。

两个端口共享同一个 FastAPI app 实例（lifespan 只跑一次），但
请求处理函数会按入站端口选择对应的 profile 强制覆盖采样参数。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load_config():
    from .config import ProxyConfig
    config_paths = [
        Path(os.getcwd()) / "config.yaml",
        _root / "config.yaml",
        Path(os.getenv("DEEPPROXY_CONFIG", "")),
    ]
    for cp in config_paths:
        if cp.exists():
            return ProxyConfig.from_yaml(cp)
    return ProxyConfig.from_env()


async def _serve_both(host: str, coding_port: int, writing_port: int, log_level: str):
    import uvicorn

    cfg_coding = uvicorn.Config(
        "deep_proxy.main:app",
        host=host,
        port=coding_port,
        log_level=log_level,
        reload=os.getenv("DEEPPROXY_RELOAD", "false").lower() == "true",
    )
    cfg_writing = uvicorn.Config(
        "deep_proxy.main:app",
        host=host,
        port=writing_port,
        log_level=log_level,
        reload=False,  # 第二个实例无法 reload（只允许一个 reload watcher）
    )
    server_coding = uvicorn.Server(cfg_coding)
    server_writing = uvicorn.Server(cfg_writing)
    await asyncio.gather(server_coding.serve(), server_writing.serve())


def main():
    """启动 DeepProxy 服务器（同时绑定 coding_port 与 writing_port）。"""
    import logging.handlers

    config = _load_config()
    log_level = config.log_level.lower()

    # 根日志器：DEBUG+（让所有消息通过）
    # 控制台 handler：INFO+（过滤掉 DEBUG）
    # 文件 handler：DEBUG+（完整记录）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # 清空默认 handler，避免重复
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # 文件：DEBUG+，完整记录（含 BERT 输入诊断 / 请求体等）
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "deepproxy.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    # -----------------------------------------------------------------------
    # LiteLLM 日志抑制（必须在 import 之前，LiteLLM 在 import 时读取环境变量）
    # -----------------------------------------------------------------------
    os.environ.setdefault("LITELLM_LOG", "WARNING")
    import litellm  # noqa: E402
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
    # 网络层 DEBUG（httpx/httpcore 等）也不进文件
    for _name in ("httpx", "httpcore", "httpcore.connection", "httpcore.http11",
                   "asyncio", "charset_normalizer", "urllib3"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    # -----------------------------------------------------------------------
    # 注册 deepseek-v4-pro / deepseek-v4-flash 到 LiteLLM 计价表
    # 消除 "This model isn't mapped yet" 错误警告
    # 价格来自 deepseek_pricing.py（$0.14/$0.28 flash, $0.435/$0.87 pro, per M tokens）
    # -----------------------------------------------------------------------
    _DEEPSEEK_PRICING: dict = {
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "litellm_provider": "deepseek",
        "mode": "chat",
    }
    litellm.model_cost.setdefault("deepseek-v4-flash", {
        **_DEEPSEEK_PRICING,
        "input_cost_per_token": 0.00000014,   # $0.14/M 输入
        "output_cost_per_token": 0.00000028,  # $0.28/M 输出
    })
    litellm.model_cost.setdefault("deepseek-v4-pro", {
        **_DEEPSEEK_PRICING,
        "input_cost_per_token": 0.000000435,  # $0.435/M 输入
        "output_cost_per_token": 0.00000087,  # $0.87/M 输出
    })

    asyncio.run(_serve_both(
        host=config.host,
        coding_port=config.coding_port,
        writing_port=config.writing_port,
        log_level=log_level,
    ))


if __name__ == "__main__":
    main()
