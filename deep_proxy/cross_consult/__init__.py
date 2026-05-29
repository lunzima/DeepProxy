"""Cross-Consult 虚拟工具 + 标签触发的整轮重定向。

两个并行机制共享 cross_consult.pairs map 与 enabled 主开关：
- 工具调用（tool-call）：agent 主动 emit cross_consult tool_call，DeepProxy 在响应路径
  拦截、向异家族 pro 模型发起一次 context-free 咨询、把结果以 tool_result 注入会话后
  重发原 provider。单次、强制 pro、不换家。
- 标签重定向（tag redirect §12.11）：DeepProxy 在请求路径扫描 user 消息中的字面标签
  `[本轮对话使用不同家族的大语言模型]`，命中则把整轮请求重路由到异家族 provider，
  并按 redirect_persist_turns 维持若干轮窗口。多轮换家、走目标家族自身分级路由。
"""
from .awareness import build_awareness_prompt
from .config import CrossConsultConfig
from .redirect import resolve_redirect
from .redirect_tracker import RedirectTracker

__all__ = [
    "CrossConsultConfig",
    "RedirectTracker",
    "build_awareness_prompt",
    "resolve_redirect",
]
