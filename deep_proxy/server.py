"""DeepProxy 服务器启动入口（绑定 config.ports 声明的全部端口）。

默认双端口（legacy / config.example.yaml）：
- coding_port (默认 8000) → precise_sampling profile
- writing_port (默认 8001) → creative_sampling profile
  写作篮在 dynamic_baskets 层按 optimization.writing_basket_kind
  （creative / general）切换；采样参数与端口数量无关。

所有端口共享同一个 FastAPI app 实例（lifespan 只跑一次），但请求处理函数会按
入站 socket 端口选择对应的 PortBinding（provider / sampling / model_pool）。
绑定的端口集合来自 config.bound_ports()（即 ports[] 声明的端口），不再硬编码
coding_port/writing_port——新格式 remap 端口时服务器随之监听正确端口。
"""

from __future__ import annotations

import asyncio
import logging.handlers  # 绑定 logging 与 logging.handlers（main 配置 RotatingFileHandler 用）
import os
import socket
import sys
from pathlib import Path

logger = logging.getLogger("deep_proxy")

# 确保项目根目录在 sys.path 中
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load_config():
    from .config import ProxyConfig
    return ProxyConfig.discover_and_load()


def _port_in_use(host: str, port: int) -> bool:
    """探测 port 是否已被监听（跨平台）。

    用 connect 探测而非 bind：Windows 下 SO_REUSEADDR/独占语义与 *nix 不同，bind 探测
    会误判；connect 成功即说明确有进程在该端口 listen。host=0.0.0.0/:: 时连环回地址。
    """
    target = "127.0.0.1" if host in ("0.0.0.0", "", "::", "0:0:0:0:0:0:0:0") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((target, port)) == 0
        except OSError:
            return False


def _assert_ports_available(host: str, ports: list[int]) -> None:
    """启动前确认端口未被占用，否则**显式报错退出**。

    历史footgun：旧 DeepProxy 进程仍持有端口时，新进程 uvicorn 静默放弃该端口（不抛错），
    旧进程继续以旧代码服务 → 改动「不生效」、反复误判。这里前置探测、一次性报清楚。
    """
    busy = [p for p in ports if _port_in_use(host, p)]
    if not busy:
        return
    busy_csv = ",".join(str(p) for p in busy)
    raise RuntimeError(
        f"端口 {busy_csv} 已被占用——极可能是旧的 DeepProxy 进程仍在运行（会以旧代码继续"
        f"服务这些端口，导致代码改动看似『不生效』）。请先结束占用进程再启动：\n"
        f"  Windows : Get-NetTCPConnection -LocalPort {busy_csv} -State Listen | "
        f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}\n"
        f"  Linux/macOS: lsof -ti tcp:{busy_csv.replace(',', ',tcp:')} | xargs kill -9"
    )


async def _serve_ports(host: str, ports: list[int], log_level: str):
    """绑定 ports[] 声明的全部端口（共享同一个 FastAPI app 实例）。

    路由按入站 socket 端口解析（main._binding_for_request），故服务器必须监听
    ports[] 声明的端口本身。自动重载在进程层由 main() 经 watchfiles 实现（见该处），
    本函数只负责绑定+服务，不再向 uvicorn 传无效的 reload（Server.serve() 不走 uvicorn
    的 reload 监视器，旧代码里的 reload=True 实为 no-op）。
    """
    import uvicorn

    _assert_ports_available(host, ports)
    servers = [
        uvicorn.Server(uvicorn.Config(
            "deep_proxy.main:app", host=host, port=port, log_level=log_level,
        ))
        for port in ports
    ]
    await asyncio.gather(*(s.serve() for s in servers))


def _setup_logging(log_level: str) -> None:
    """根日志器 DEBUG+；控制台 handler 按 log_level；文件 handler DEBUG 全量。

    清空已有 handler 后重建，确保 reload 子进程每次启动幂等（不重复挂 handler）。
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
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
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)


def _register_litellm_pricing() -> None:
    """抑制 LiteLLM 噪声日志 + 注册 deepseek / MiMo 计价，消除 'model isn't mapped' 警告。"""
    # LiteLLM 日志抑制（必须在 import 之前，LiteLLM 在 import 时读取环境变量）
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

    # deepseek-v4-pro / deepseek-v4-flash → LiteLLM 计价表（价格统一来自 deepseek_pricing.py）
    from .deepseek_pricing import _DEEPSEEK_PRICING as _DP, _V4_CONTEXT_WINDOW, _V4_MAX_OUTPUT
    for _model_name, _p in _DP.items():
        litellm.model_cost.setdefault(_model_name, {
            "max_tokens": _V4_MAX_OUTPUT,
            "max_input_tokens": _V4_CONTEXT_WINDOW,
            "max_output_tokens": _V4_MAX_OUTPUT,
            "litellm_provider": "deepseek",
            "mode": "chat",
            "input_cost_per_token": _p["prompt"] / 1_000_000,
            "output_cost_per_token": _p["completion"] / 1_000_000,
        })

    # MiMo 模型 → LiteLLM 计价表（走 openai/ 前缀，与 deepseek 注册相互独立）
    from .mimo_pricing import (
        _MIMO_PRICING as _MP,
        _MIMO_CONTEXT_WINDOW as _MIMO_CTX,
        _MIMO_MAX_OUTPUT as _MIMO_MXO,
    )
    for _model_name, _p in _MP.items():
        full_name = f"openai/{_model_name}"
        litellm.model_cost.setdefault(full_name, {
            "max_tokens": _MIMO_MXO,
            "max_input_tokens": _MIMO_CTX,
            "max_output_tokens": _MIMO_MXO,
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": _p["prompt"] / 1_000_000,
            "output_cost_per_token": _p["completion"] / 1_000_000,
        })


def _run_server() -> None:
    """完整阻塞式启动：日志 + 计价注册 + 绑定端口 serve。

    直跑（无 reload）与 reload 子进程共用此入口；reload 子进程每次重启都会重跑全套
    setup（全新进程 → import 最新代码），故改动确定生效。
    """
    config = _load_config()
    log_level = config.log_level.lower()
    _setup_logging(log_level)
    _register_litellm_pricing()
    asyncio.run(_serve_ports(
        host=config.host,
        ports=config.bound_ports(),
        log_level=log_level,
    ))


def main():
    """启动 DeepProxy 服务器（绑定 config.ports 声明的全部端口）。

    DEEPPROXY_RELOAD=true 时经 watchfiles.run_process 真正实现自动重载：父进程监视
    deep_proxy/ 包目录，任何 .py 改动 → 终止并重启运行 _run_server 的子进程（整进程
    重启 → 全端口随新代码起来）。只监视包目录、不监视 logs/，避免日志写入触发重载死循环。
    config.yaml 改动不在监视范围（需手动重启）。
    """
    reload_enabled = os.getenv("DEEPPROXY_RELOAD", "false").lower() == "true"
    if not reload_enabled:
        _run_server()
        return

    from watchfiles import run_process

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [deep_proxy.reload] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    pkg_dir = str(Path(__file__).parent)
    logging.getLogger("deep_proxy.reload").info(
        "DEEPPROXY_RELOAD=true：监视 %s，.py 改动将重启全部端口（config.yaml 改动需手动重启）",
        pkg_dir,
    )
    run_process(pkg_dir, target=_run_server)


if __name__ == "__main__":
    main()
