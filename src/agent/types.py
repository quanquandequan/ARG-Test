"""Agent-related data types."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.types import ToolCall


@dataclass
class AgentStep:
    step_index: int
    tool_call: ToolCall | None = None
    tool_result: str = ""
    thinking: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    citations: list[dict] = field(default_factory=list)
