"""Test point domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(slots=True)
class TestPoint:
    """A testable focus derived from RequirementIR."""

    __test__: ClassVar[bool] = False

    id: str
    title: str
    feature_id: str = ""
    priority: str = "P1"
    test_type: str = "功能"
    risk_level: str = "medium"
    source: str = ""
    hints: list[str] = field(default_factory=list)
