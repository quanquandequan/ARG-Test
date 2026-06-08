"""Base workflow node contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.domain.artifacts.test_design_artifact import TestDesignArtifact
from src.domain.requirement import RequirementIR
from src.domain.requirements import GeneratedTestCase, TestCaseGenerationRequest
from src.domain.test_design import TestPoint, TestScenario


@dataclass(slots=True)
class WorkflowContext:
    request: TestCaseGenerationRequest
    requirement_text: str
    module: str
    generation_mode: str
    kb_samples: str = ""
    requirement_ir: RequirementIR | None = None
    test_points: list[TestPoint] = field(default_factory=list)
    scenarios: list[TestScenario] = field(default_factory=list)
    test_cases: list[GeneratedTestCase] = field(default_factory=list)
    artifact: TestDesignArtifact | None = None


class WorkflowNode(ABC):
    """A deterministic workflow step that reads/writes WorkflowContext."""

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        ...
