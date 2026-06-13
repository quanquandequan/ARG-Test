"""测试用例生成流程的请求/结果 DTO。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from src.domain.artifacts import ArtifactRecord

if TYPE_CHECKING:
    from src.domain.artifacts.test_design_artifact import TestDesignArtifact

from src.domain.test_design.generated_test_case import GeneratedTestCase


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
    artifact: "TestDesignArtifact | None" = None

    @property
    def case_count(self) -> int:
        return len(self.cases)


@dataclass(slots=True)
class TestCaseGenerationResult:
    generation: TestCaseGenerationData
    workbook_artifact: ArtifactRecord | None = None
    automation_json_artifact: ArtifactRecord | None = None
    summary: str = ""
