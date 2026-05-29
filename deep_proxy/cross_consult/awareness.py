"""双家族机制 awareness — system prompt 状态披露。

向 agent 披露本代理桥接两个异家族 LLM 的事实，并说明三条可选切换路径：
保持现状 / 调用 cross_consult 工具 / 在 user 消息插入字面标签触发整轮重定向。

刻意不暴露的实现细节（plan §2）：
- 不提 BERT / router / heuristic / score / threshold / persist_turns 等内部参数
- 不描述目标家族会用 flash 还是 pro 档，只说"按任务自适应"
- 不暴露 max_calls_per_request 之外的内部计数器

文案随实现演进的耐受度由"模糊化"换取，避免 agent 针对阈值做 prompt engineering。
"""
from __future__ import annotations


_REDIRECT_TAG_LITERAL = "[本轮对话使用不同家族的大语言模型]"


def build_awareness_prompt(
    *,
    source_provider_name: str,
    target_provider_name: str,
    tool_name: str = "cross_consult",
    max_calls: int = 3,
) -> str:
    """构造追加到 system prompt 末尾的双家族 awareness 段。

    Args:
        source_provider_name: 当前请求实际去往的 provider 名（已应用任何重定向）。
        target_provider_name: 异家族对偶 provider 名。
        tool_name: cross_consult 工具的暴露名。
        max_calls: 单次请求 cross_consult 调用上限（这是允许暴露给 agent 的少数实数之一）。

    Returns:
        约 10-14 行的中文段落，前置两个换行确保与既有 system prompt 隔行；
        末尾不带换行（由下游 addendum 衔接）。
    """
    source = source_provider_name
    target = target_provider_name
    return (
        f"\n\n[DeepProxy 双家族披露] 本代理桥接两个异家族大语言模型：当前会话由 `{source}` 家族承担，"
        f"对偶家族为 `{target}`。两者训练分布不同，擅长面相互补。\n"
        f"你有三种选择，按任务自行权衡：\n"
        f"1. 默认：保持在 `{source}` 家族——多数场景已足够。\n"
        f"2. 单点借援：调用工具 `{tool_name}` 向 `{target}` 家族请求第二视角（"
        f"本次会话最多 {max_calls} 次；目标无本次上下文，question 需 self-contained）。"
        f"该路径**单次、强制使用对方较高层级模型**，适合"
        f"独立子问题、二次验证、打破认知惯性。\n"
        f"3. 多轮次换家：在 user 消息任意位置插入字面标签 `{_REDIRECT_TAG_LITERAL}`，"
        f"代理将把当前轮及随后若干轮重路由到 `{target}` 家族，由对方家族**按任务自适应选择适当层级**"
        f"承接完整对话上下文。窗口耗尽后对话自然回到 `{source}` 家族；如需继续换家请按需重新插入标签。\n"
        f"标签必须严格使用上述字面形式才会触发；不要改写或意译。"
    )
