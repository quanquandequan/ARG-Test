"""Conversation history management — sliding-window truncation with token budget.

The truncation algorithm:
1. Always keep the system message (index 0).
2. Always keep the last N turns of the live conversation (``keep_last``).
3. For every remaining history message, estimate its token count as
   ``len(content) // 4`` (a conservative rough estimate that avoids a real
   tokeniser dependency).
4. Fill backwards from the oldest kept turn until ``max_history_tokens`` is
   exhausted.

This gives a predictable context length without needing a provider-specific
tokeniser.
"""

from __future__ import annotations

from src.llm.types import Message

# Approximate tokens per character for Chinese/English mixed text.
# Chinese ≈ 1.5 chars/token, English ≈ 4 chars/token → ~2.5 chars/token average.
_CHARS_PER_TOKEN: float = 2.5

_DEFAULT_MAX_HISTORY_TOKENS = 4000
_DEFAULT_KEEP_LAST = 4  # always keep the last 4 messages unconditionally


def _estimate_tokens(msg: Message) -> int:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return max(1, int(len(content) / _CHARS_PER_TOKEN))


def truncate_history(
    history: list[Message],
    max_tokens: int = _DEFAULT_MAX_HISTORY_TOKENS,
    keep_last: int = _DEFAULT_KEEP_LAST,
) -> list[Message]:
    """Return a token-budget-aware slice of ``history``.

    ``history`` must NOT include the system prompt or the current user query;
    those are handled by the caller.
    """
    if not history:
        return []

    # Always keep the most recent ``keep_last`` messages unconditionally.
    keep_last = min(keep_last, len(history))
    unconditional = history[-keep_last:] if keep_last else []
    older = history[:-keep_last] if keep_last < len(history) else []

    # Budget is reduced by what the unconditional tail already consumes.
    remaining = max_tokens - sum(_estimate_tokens(m) for m in unconditional)
    if remaining <= 0:
        return list(unconditional)

    # Fill greedily from the right (newest first) of the older slice.
    selected: list[Message] = []
    for msg in reversed(older):
        cost = _estimate_tokens(msg)
        if remaining - cost < 0:
            break
        selected.insert(0, msg)
        remaining -= cost

    return selected + list(unconditional)
