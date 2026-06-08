"""需求与测试用例工作流的类型化 DTO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from src.domain.artifacts import ArtifactRecord

if TYPE_CHECKING:
    from src.domain.artifacts.test_design_artifact import TestDesignArtifact


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
    data_setup: str = ""
    selectors: str = ""
    automation_steps: str = ""
    assertions: str = ""
    business_name: str = ""
    ui_display_name: str = ""
    page_route: str = ""
    locator_chain: str = ""
    anchor_text: str = ""
    search_strategy: str = ""
    expected_visibility: str = ""
    forbidden_locators: str = ""


@dataclass(slots=True)
class TestCaseGenerationRequest:
    __test__: ClassVar[bool] = False

    requirement: str
    kb_samples: str = ""
    module: str = ""
    output_dir: str = ""
    generation_mode: str = "manual"
    system_prompt_override: str = ""
    request_id: str = ""


@dataclass(slots=True)
class TestCaseGenerationData:
    module: str
    cases: list[GeneratedTestCase]
    kb_samples: str = ""
    generation_mode: str = "manual"
    artifact: TestDesignArtifact | None = None

    @property
    def case_count(self) -> int:
        return len(self.cases)


@dataclass(slots=True)
class TestCaseGenerationResult:
    generation: TestCaseGenerationData
    workbook_artifact: ArtifactRecord
    automation_json_artifact: ArtifactRecord | None = None
    summary: str = ""
