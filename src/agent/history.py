"""对话历史管理：按 token 预算做滑动窗口截断。

截断算法：
1. 始终保留 system 消息（索引 0）。
2. 始终保留实时对话的最后 N 轮（``keep_last``）。
3. 对剩余历史消息，按 ``len(content) // 4`` 估算 token 数
   （保守粗略估算，避免依赖真实 tokenizer）。
4. 从已保留轮次之前开始向前填充，直到 ``max_history_tokens`` 用完。

这样无需 provider 专用 tokenizer，也能得到可预测的上下文长度。
"""

from __future__ import annotations

from src.llm.types import Message

# 中英文混合文本中每个 token 对应的近似字符数。
# 中文约 1.5 字符/token，英文约 4 字符/token，平均约 2.5 字符/token。
_CHARS_PER_TOKEN: float = 2.5

_DEFAULT_MAX_HISTORY_TOKENS = 4000
_DEFAULT_KEEP_LAST = 4  # 始终无条件保留最后 4 条消息


def _estimate_tokens(msg: Message) -> int:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return max(1, int(len(content) / _CHARS_PER_TOKEN))


def truncate_history(
    history: list[Message],
    max_tokens: int = _DEFAULT_MAX_HISTORY_TOKENS,
    keep_last: int = _DEFAULT_KEEP_LAST,
) -> list[Message]:
    """返回符合 token 预算的 ``history`` 切片。

    ``history`` 不应包含 system prompt 或当前用户问题；这些由调用方处理。
    """
    if not history:
        return []

    # 始终无条件保留最近的 ``keep_last`` 条消息。
    keep_last = min(keep_last, len(history))
    unconditional = history[-keep_last:] if keep_last else []
    older = history[:-keep_last] if keep_last < len(history) else []

    # 扣除无条件保留尾部已消耗的预算。
    remaining = max_tokens - sum(_estimate_tokens(m) for m in unconditional)
    if remaining <= 0:
        return list(unconditional)

    # 从旧消息切片右侧（较新的消息）开始贪心填充。
    selected: list[Message] = []
    for msg in reversed(older):
        cost = _estimate_tokens(msg)
        if remaining - cost < 0:
            break
        selected.insert(0, msg)
        remaining -= cost

    return selected + list(unconditional)
