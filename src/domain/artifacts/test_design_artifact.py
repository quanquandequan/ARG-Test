"""Unified test design artifact consumed by exporters and future executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.domain.requirement import RequirementIR
from src.domain.requirements import GeneratedTestCase
from src.domain.test_design.test_point import TestPoint
from src.domain.test_design.test_scenario import TestScenario


@dataclass(slots=True)
class TestDesignArtifact:
    """Full intermediate asset produced by test-case generation workflow."""

    __test__: ClassVar[bool] = False

    module: str
    generation_mode: str
    requirement_ir: RequirementIR
    test_points: list[TestPoint] = field(default_factory=list)
    scenarios: list[TestScenario] = field(default_factory=list)
    test_cases: list[GeneratedTestCase] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.test_cases)
