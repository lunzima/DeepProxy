"""标签重定向的对话级持久化跟踪器。

仿 `optimization.flash_upgrade.UpgradeTracker` 模式，把"当前对话剩余 N 轮重定向"
保存在 in-memory OrderedDict 中。key 复用 utils.conversation_fingerprint +
utils.last_user_hash（同 UpgradeTracker 一致，确保两个 tracker 对"新轮次"的
判定标准一致）。

语义差异：
- UpgradeTracker 跟踪"flash→pro 升格"的 N 轮持续
- RedirectTracker 跟踪"源 provider→目标 provider 重定向"的 N 轮持续
两者计数互不干扰，独立维护。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from ..utils import conversation_fingerprint, last_user_hash


class RedirectTracker:
    """按对话指纹跟踪"标签触发的整轮 provider 重定向"剩余轮数。

    key = (fingerprint, last_user_hash, source_provider_name)
    value = remaining_turns（含当前轮）

    "新轮次" 通过 last_user_hash 变化检测（同 UpgradeTracker），
    保证客户端做对话压缩 / 历史合并时计数仍稳健。

    source_provider_name 是 *入站* port 绑定的 provider 名（即用户配置原本要去的
    家族），不是重定向后的目标。这样同一对话在不同 port 触发重定向时互不干扰。
    """

    def __init__(self, max_size: int = 512):
        self._sessions: OrderedDict[Tuple[str, str, str], int] = OrderedDict()
        self._max = max_size

    def consume_turn(
        self, messages: List[Dict[str, Any]], *, source_provider_name: str,
    ) -> Tuple[bool, int]:
        """当前对话是否处于重定向窗口内，并返回剩余轮数。

        副作用：检测到新轮次（last_user_hash 变化）时消耗 1 轮额度并迁移 key。

        Returns:
            (active, remaining)：active=True 表示本轮应继续重定向；
            remaining = 含本轮的剩余总轮数（active=False 时为 0）。
        """
        fp = conversation_fingerprint(messages)
        current_hash = last_user_hash(messages)

        # 当前 hash 已记账
        key = (fp, current_hash, source_provider_name)
        if key in self._sessions:
            remaining = self._sessions[key]
            if remaining <= 0:
                del self._sessions[key]
                return False, 0
            return True, remaining

        # 同 fp + source 但 hash 不同 → 新轮次，扣减后迁移到新 hash
        stale_key = None
        for k in list(self._sessions.keys()):
            if k[0] == fp and k[2] == source_provider_name and k[1] != current_hash:
                stale_key = k
                break
        if stale_key is None:
            return False, 0

        remaining = self._sessions[stale_key] - 1
        del self._sessions[stale_key]
        if remaining <= 0:
            return False, 0
        new_key = (fp, current_hash, source_provider_name)
        self._sessions[new_key] = remaining
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return True, remaining

    def is_redirected(
        self, messages: List[Dict[str, Any]], *, source_provider_name: str,
    ) -> bool:
        """consume_turn 的布尔便捷封装，保留向后兼容。新代码直接用 consume_turn。"""
        active, _ = self.consume_turn(
            messages, source_provider_name=source_provider_name,
        )
        return active

    def set_remaining(
        self, messages: List[Dict[str, Any]], turns: int,
        *, source_provider_name: str,
    ) -> None:
        """写入剩余轮次。turns 含当前请求；turns=1 表示"仅本次"。"""
        key = (
            conversation_fingerprint(messages),
            last_user_hash(messages),
            source_provider_name,
        )
        self._sessions[key] = turns
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    def remaining(
        self, messages: List[Dict[str, Any]], *, source_provider_name: str,
    ) -> int:
        """只读查询剩余轮次。"""
        key = (
            conversation_fingerprint(messages),
            last_user_hash(messages),
            source_provider_name,
        )
        return self._sessions.get(key, 0)

    def clear(
        self, messages: List[Dict[str, Any]], *, source_provider_name: str,
    ) -> None:
        """主动清除当前对话的重定向状态。"""
        key = (
            conversation_fingerprint(messages),
            last_user_hash(messages),
            source_provider_name,
        )
        self._sessions.pop(key, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)
