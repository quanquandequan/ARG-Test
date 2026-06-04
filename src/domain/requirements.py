"""Typed DTOs for requirements and test-case workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.artifacts import ArtifactRecord


@dataclass(slots=True)
class RequirementAnalysisData:
    module: str
    summary: str
    graph: dict[str, Any]
    feature_count: int
    risk_count: int
    clarification_count: int
    kb_context: str = ""


@dataclass(slots=True)
class RequirementAnalysisResult:
    analysis: RequirementAnalysisData
    json_artifact: ArtifactRecord
    markdown_artifact: ArtifactRecord


@dataclass(slots=True)
class GeneratedTestCase:
    title: str
    module: str
    precondition: str
    steps: str
    expected: str
    priority: str
    case_type: str
    notes: str = ""


@dataclass(slots=True)
class TestCaseGenerationData:
    module: str
    cases: list[GeneratedTestCase]
    kb_samples: str = ""

    @property
    def case_count(self) -> int:
        return len(self.cases)


@dataclass(slots=True)
class TestCaseGenerationResult:
    generation: TestCaseGenerationData
    workbook_artifact: ArtifactRecord

