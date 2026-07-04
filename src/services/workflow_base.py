"""Base workflow node contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.domain.artifacts.test_design_artifact import TestDesignArtifact
from src.domain.requirement import RequirementIR
from src.domain.test_design.generated_test_case import GeneratedTestCase
from src.domain.test_design.generation import TestCaseGenerationRequest
from src.domain.test_design import TestPoint, TestScenario


@dataclass(slots=True)
class WorkflowContext:
    request: TestCaseGenerationRequest
    requirement_text: str
    module: str
    generation_mode: str
    kb_samples: str = ""
    # 新增功能影响面：来自需求分析 regression_scope 的知识库现状检索结果，
    # 用于让用例生成器为受影响的既有页面/模块补充回归验证用例。
    regression_context: str = ""
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
