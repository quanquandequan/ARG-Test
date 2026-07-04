"""ExecuteScenarioTool 单条/批量分发逻辑单元测试。

回归背景：新增 case_ids/max_cases/exclude_types 批量参数后，工具需要正确
区分"单条执行"（历史行为，调用 workflow.execute）和"批量执行"（调用
workflow.execute_batch 并把汇总结果格式化成摘要文本），且不能破坏未传任何
批量参数时的既有单条执行行为。
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.agent.tools.execute_scenario import ExecuteScenarioTool
from src.domain.execution import ScenarioBatchExecutionResult, ScenarioExecutionResult


def _make_automation_json(tmp_dir: str) -> str:
    payload = {"module": "测试模块", "cases": [{"id": "C001", "title": "t1"}]}
    path = Path(tmp_dir) / "automation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestExecuteScenarioToolDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_no_batch_params_calls_single_case_execute(self):
        workflow = MagicMock()
        workflow.execute = AsyncMock(
            return_value=ScenarioExecutionResult(
                case_id="C001", title="t1", module="测试模块", status="PASS"
            )
        )
        workflow.execute_batch = AsyncMock()
        tool = ExecuteScenarioTool(workflow=workflow)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_automation_json(tmp)
            result = await tool.execute(automation_json_path=path)

        workflow.execute.assert_awaited_once()
        workflow.execute_batch.assert_not_awaited()
        self.assertIn("执行结果：PASS", result)

    async def test_max_cases_param_routes_to_batch_execution(self):
        workflow = MagicMock()
        workflow.execute = AsyncMock()
        workflow.execute_batch = AsyncMock(
            return_value=ScenarioBatchExecutionResult(
                module="测试模块",
                results=[
                    ScenarioExecutionResult(
                        case_id="C001", title="t1", module="测试模块", status="PASS"
                    ),
                    ScenarioExecutionResult(
                        case_id="C002",
                        title="t2",
                        module="测试模块",
                        status="FAIL",
                        failure_reason="断言失败",
                    ),
                ],
                skipped_case_ids=["C003"],
            )
        )
        tool = ExecuteScenarioTool(workflow=workflow)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_automation_json(tmp)
            result = await tool.execute(automation_json_path=path, max_cases=2)

        workflow.execute_batch.assert_awaited_once()
        workflow.execute.assert_not_awaited()
        self.assertIn("批量执行完成", result)
        self.assertIn("通过 1 条，失败 1 条", result)
        self.assertIn("C001", result)
        self.assertIn("C002", result)
        self.assertIn("断言失败", result)
        self.assertIn("C003", result)  # 跳过的用例也要如实告知

    async def test_exclude_types_param_also_routes_to_batch(self):
        workflow = MagicMock()
        workflow.execute = AsyncMock()
        workflow.execute_batch = AsyncMock(
            return_value=ScenarioBatchExecutionResult(module="测试模块", results=[])
        )
        tool = ExecuteScenarioTool(workflow=workflow)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_automation_json(tmp)
            await tool.execute(automation_json_path=path, exclude_types=["回归测试"])

        workflow.execute_batch.assert_awaited_once()
        workflow.execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
