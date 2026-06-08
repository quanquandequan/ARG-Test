"""Execution intent models reserved for future UI automation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ExecutionAction:
    """One intended executable action derived from a scenario."""

    action: str
    target: str = ""
    value: str = ""
    locator_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionPlan:
    """Automation-ready intent that can later feed MobileExecutionWorkflow."""

    actions: list[ExecutionAction] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    locator_hints: list[str] = field(default_factory=list)
