"""Typed tool result wrapper with an agent-friendly text rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.artifacts import ArtifactRecord


@dataclass(slots=True)
class ToolExecutionResult:
    content: str
    data: Any = None
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content

