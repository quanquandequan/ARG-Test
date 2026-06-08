"""Test point generator workflow node."""

from __future__ import annotations

from src.application.workflow_nodes.base import WorkflowContext, WorkflowNode
from src.domain.test_design import TestPoint


class TestPointGeneratorNode(WorkflowNode):
    """Derive test points from RequirementIR features and business rules."""

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        ir = context.requirement_ir
        if ir is None:
            raise ValueError("RequirementIR is required before generating test points.")

        points: list[TestPoint] = []
        for idx, feature in enumerate(ir.features, start=1):
            points.append(
                TestPoint(
                    id=f"TP{idx:03d}",
                    title=feature.name,
                    feature_id=feature.id,
                    priority=feature.priority,
                    test_type="功能",
                    risk_level="medium",
                    source=f"feature:{feature.id}",
                    hints=list(feature.test_hints),
                )
            )
        offset = len(points)
        for idx, rule in enumerate(ir.business_rules, start=1):
            points.append(
                TestPoint(
                    id=f"TP{offset + idx:03d}",
                    title=rule.description,
                    priority="P1",
                    test_type="规则",
                    risk_level="medium",
                    source=f"rule:{rule.id}",
                    hints=[rule.condition, rule.outcome],
                )
            )
        if not points:
            points.append(
                TestPoint(
                    id="TP001",
                    title=context.module,
                    priority="P0",
                    test_type="功能",
                    source="requirement",
                )
            )
        context.test_points = points
        return context
