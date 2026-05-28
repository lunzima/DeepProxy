"""Cross-Consult 虚拟工具：让任意 port 临时调用异家族 pro 模型。

provider 之间通过 `pairs` map 对称声明对偶关系；DeepProxy 在响应路径拦截
`cross_consult` tool_use 并代为执行，结果以 tool_result 注入会话后重发原 provider。
"""
from .config import CrossConsultConfig

__all__ = ["CrossConsultConfig"]
