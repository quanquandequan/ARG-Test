"""Test scenario domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from src.domain.test_design.execution_plan import ExecutionPlan


@dataclass(slots=True)
class TestScenario:
    """Core test design asset that can drive case export or execution."""

    __test__: ClassVar[bool] = False

    id: str
    title: str
    point_id: str = ""
    precondition: str = "无"
    steps_intent: list[str] = field(default_factory=list)
    expected_intent: list[str] = field(default_factory=list)
    data_state: str = "normal"
    priority: str = "P1"
    test_type: str = "功能"
    execution_intent: ExecutionPlan = field(default_factory=ExecutionPlan)
