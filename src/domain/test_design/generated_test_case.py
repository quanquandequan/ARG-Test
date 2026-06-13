"""生成的测试用例 DTO。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    selectors: list[str] = field(default_factory=list)
    automation_steps: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    business_name: str = ""
    ui_display_name: str = ""
    page_route: str = ""
    locator_chain: str = ""
    anchor_text: str = ""
    search_strategy: str = ""
    expected_visibility: str = ""
    forbidden_locators: list[str] = field(default_factory=list)


__all__ = ["GeneratedTestCase"]
