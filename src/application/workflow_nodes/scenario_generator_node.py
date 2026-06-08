"""Scenario generator workflow node."""

from __future__ import annotations

from src.application.workflow_nodes.base import WorkflowContext, WorkflowNode
from src.domain.test_design import ExecutionAction, ExecutionPlan, TestScenario


class ScenarioGeneratorNode(WorkflowNode):
    """Build scenario assets from test points."""

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        scenarios: list[TestScenario] = []
        for idx, point in enumerate(context.test_points, start=1):
            plan = ExecutionPlan(
                actions=[
                    ExecutionAction(
                        action="verify",
                        target=point.title,
                        locator_hints=[point.title],
                    )
                ],
                assertions=[point.title],
                locator_hints=[point.title],
            )
            scenarios.append(
                TestScenario(
                    id=f"SC{idx:03d}",
                    title=point.title,
                    point_id=point.id,
                    precondition="无",
                    steps_intent=[f"验证 {point.title}"],
                    expected_intent=[f"{point.title} 符合需求"],
                    data_state="normal",
                    priority=point.priority,
                    test_type=point.test_type,
                    execution_intent=plan,
                )
            )
        context.scenarios = scenarios
        return context
