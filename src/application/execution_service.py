"""移动端自动化执行服务。"""

from __future__ import annotations

from src.application.workflows.execution_workflow import ExecutionWorkflow
from src.domain.execution import ScenarioExecutionRequest, ScenarioExecutionResult


class MobileExecutionService:
    """对外提供自动化场景执行能力。"""

    def __init__(self, workflow: ExecutionWorkflow):
        self._workflow = workflow

    async def execute_scenario(
        self,
        request: ScenarioExecutionRequest,
    ) -> ScenarioExecutionResult:
        return await self._workflow.execute(request)
