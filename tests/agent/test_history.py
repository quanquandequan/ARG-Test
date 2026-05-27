"""Tests for history truncation logic."""

from src.agent.history import _estimate_tokens, truncate_history
from src.llm.types import Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


class TestEstimateTokens:
    def test_empty_content(self):
        # minimum 1
        assert _estimate_tokens(_msg("user", "")) == 1

    def test_ascii_text(self):
        # 40 chars / 2.5 ≈ 16
        m = _msg("user", "a" * 40)
        assert _estimate_tokens(m) == 16

    def test_cjk_text(self):
        # 10 CJK chars / 2.5 = 4
        m = _msg("user", "你好世界" * 3)  # 12 chars → 4.8 → 4
        assert _estimate_tokens(m) == 4


class TestTruncateHistory:
    def test_empty_history_returns_empty(self):
        assert truncate_history([]) == []

    def test_short_history_kept_intact(self):
        msgs = [_msg("user", "A"), _msg("assistant", "B")]
        result = truncate_history(msgs, max_tokens=1000, keep_last=2)
        assert result == msgs

    def test_keep_last_always_preserved(self):
        # Budget forces dropping of older messages
        msgs = [_msg("user", "X" * 500), _msg("user", "Y"), _msg("assistant", "Z")]
        result = truncate_history(msgs, max_tokens=10, keep_last=2)
        # Last 2 must be there
        assert result[-2].content == "Y"
        assert result[-1].content == "Z"

    def test_older_messages_dropped_when_budget_exceeded(self):
        old = [_msg("user", "OLD " * 200) for _ in range(5)]  # 800 chars each
        recent = [_msg("user", "A"), _msg("assistant", "B")]
        msgs = old + recent
        result = truncate_history(msgs, max_tokens=50, keep_last=2)
        # Old messages should not appear (each costs ~320 tokens)
        assert all("OLD" not in m.content for m in result)
        assert len(result) == 2  # only the recent pair

    def test_all_fit_nothing_dropped(self):
        msgs = [_msg("user", "short"), _msg("assistant", "ok")] * 3
        result = truncate_history(msgs, max_tokens=10000, keep_last=2)
        assert result == msgs

    def test_keep_last_clamps_to_list_length(self):
        msgs = [_msg("user", "A")]
        result = truncate_history(msgs, max_tokens=1000, keep_last=100)
        assert result == msgs
