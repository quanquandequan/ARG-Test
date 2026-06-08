"""Agent 相关数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.types import ToolCall


@dataclass
class Citation:
    index: int


@dataclass
class AgentStep:
    step_index: int
    tool_call: ToolCall | None = None
    tool_result: str = ""
    thinking: str = ""
    duration_ms: float = 0.0  # 此步骤的墙钟耗时（LLM + 工具执行）


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    citations: list[Citation] = field(default_factory=list)
    processing_stages: dict[str, float] = field(default_factory=dict)
    trace_id: str = ""
