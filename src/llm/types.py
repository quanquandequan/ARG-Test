"""Message and tool-use types for multi-turn chat."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ContentBlock:
    """A streaming content block — currently only text is used."""

    type: str  # "text"
    text: str = ""


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str | list[ContentBlock] = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ChatResponse:
    content: str
    model: str
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", "max_tokens", "stop_sequence"
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
