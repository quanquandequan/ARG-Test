"""ExecutionWorkflow 会话探活重连逻辑单元测试。

回归背景：CLI 长会话里，Appium 服务端可能因 newCommandTimeout 到期等原因
已经销毁会话，但 AppiumDriverManager 本地仍持有非空的 driver 引用。
ExecutionWorkflow.execute() 之前用 ``is_connected()``（只判断引用是否非空）
决定是否跳过重连，导致这种情况下误判"已连接"、跳过重连，后续步骤直接
抛 ``NoSuchDriverError``。改为 ``probe_session_alive()`` 主动探活后，
探测到会话已死会自动清空引用并触发重新 connect。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.core.config import load_config
from src.domain.execution import ScenarioExecutionRequest
from src.workflows.execution import ExecutionWorkflow

load_config()


def _make_payload(automation_steps: list[dict] | None = None) -> dict:
    return {
        "module": "测试模块",
        "cases": [
            {
                "id": "C001",
                "title": "示例用例",
                "automation_steps": automation_steps or [],
                "assertions": [],
            }
        ],
    }


def _make_workflow(driver_manager) -> ExecutionWorkflow:
    artifacts = MagicMock()
    artifacts.save_json.return_value = MagicMock(path="report.json")
    return ExecutionWorkflow(
        driver_manager=driver_manager,
        page_cache=MagicMock(),
        artifacts=artifacts,
    )


class TestExecutionWorkflowSessionReconnect(unittest.IsolatedAsyncioTestCase):
    async def test_reconnects_when_stale_session_detected(self):
        # probe_session_alive 返回 False 模拟"本地引用非空但服务端会话已死"，
        # 此时应该走重新 connect 分支。
        driver_manager = MagicMock()
        driver_manager.probe_session_alive = AsyncMock(return_value=False)
        driver_manager.is_connected = MagicMock(return_value=False)
        driver_manager.connect = AsyncMock()
        driver_manager.get_current_activity = AsyncMock(return_value="MainActivity")

        workflow = _make_workflow(driver_manager)
        request = ScenarioExecutionRequest(automation_json_path="dummy.json")

        # 用一个空 automation_steps 的 payload，让 execute 在 connect 之后
        # 立刻因缺少步骤而结束，从而只聚焦验证是否触发了重连。
        import json as _json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(_json.dumps(_make_payload()), encoding="utf-8")
            request.automation_json_path = str(path)
            result = await workflow.execute(request)

        driver_manager.connect.assert_awaited_once()
        self.assertEqual(result.status, "FAIL")
        self.assertIn("缺少 automation_steps", result.failure_reason)

    async def test_skips_reconnect_when_session_alive(self):
        # probe_session_alive 返回 True，说明会话仍然存活，不应该再次 connect。
        driver_manager = MagicMock()
        driver_manager.probe_session_alive = AsyncMock(return_value=True)
        driver_manager.is_connected = MagicMock(return_value=False)
        driver_manager.connect = AsyncMock()

        workflow = _make_workflow(driver_manager)

        import json as _json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(_json.dumps(_make_payload()), encoding="utf-8")
            request = ScenarioExecutionRequest(automation_json_path=str(path))
            result = await workflow.execute(request)

        driver_manager.connect.assert_not_awaited()
        self.assertEqual(result.status, "FAIL")
        self.assertIn("缺少 automation_steps", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
