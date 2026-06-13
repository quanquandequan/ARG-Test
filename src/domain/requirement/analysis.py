"""需求分析结果 DTO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RequirementAnalysisData:
    module: str
    summary: str
    graph: dict[str, Any]
    feature_count: int
    risk_count: int
    clarification_count: int
    kb_context: str = ""
