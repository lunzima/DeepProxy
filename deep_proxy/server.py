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
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = _load_config()

    asyncio.run(_serve_both(
        host=config.host,
        coding_port=config.coding_port,
        writing_port=config.writing_port,
        log_level=log_level,
    ))


if __name__ == "__main__":
    main()
