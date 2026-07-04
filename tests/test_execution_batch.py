"""ExecutionWorkflow 批量执行（execute_batch / _select_cases）单元测试。

回归背景：用户一次性传入一个包含数十条 case 的自动化 JSON，要求"只跑前 N 条，
回归测试的 case 不需要执行"。此前 execute_scenario 一次只能跑一条（未指定
case_id/case_title 时固定执行 cases[0]），且硬路由只透传路径、Agent 调用一次
工具后就结束当前轮次，导致这类批量指令无法被正确执行。现在
ExecutionWorkflow.execute_batch() 支持按 max_cases/exclude_types/case_ids
一次性选出并顺序执行多条用例，某条失败不会中断后续用例。
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.core.config import load_config
from src.domain.execution import ScenarioExecutionRequest
from src.workflows.execution import ExecutionWorkflow, _select_cases

load_config()


def _make_case(case_id: str, case_type: str) -> dict:
    return {
        "id": case_id,
        "title": f"{case_id}-标题",
        "type": case_type,
        # 空 automation_steps 会让 _run_case 立刻以 FAIL 结束，
        # 不需要 mock 完整的 action/screen/assertion 工具链即可验证批量编排逻辑。
        "automation_steps": [],
        "assertions": [],
    }


def _make_payload() -> dict:
    return {
        "module": "测试模块",
        "cases": [
            _make_case("C001", "UI测试"),
            _make_case("C002", "交互测试"),
            _make_case("C003", "回归测试"),
            _make_case("C004", "功能测试"),
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


class TestSelectCases(unittest.TestCase):
    def test_exclude_types_filters_before_max_cases_slicing(self):
        # "前 2 条非回归用例" 应该是过滤掉回归测试后再取前 2 条，
        # 而不是先取 JSON 里前 2 个位置再筛掉回归。
        payload = _make_payload()
        selected, skipped = _select_cases(
            payload, case_ids=[], max_cases=2, exclude_types=["回归测试"]
        )
        self.assertEqual([c["id"] for c in selected], ["C001", "C002"])
        self.assertEqual(skipped, ["C003"])

    def test_explicit_case_ids_selected_in_given_order(self):
        payload = _make_payload()
        selected, skipped = _select_cases(
            payload, case_ids=["C004", "C001"], max_cases=None, exclude_types=[]
        )
        self.assertEqual([c["id"] for c in selected], ["C004", "C001"])
        self.assertEqual(skipped, [])

    def test_unknown_case_id_recorded_as_skipped(self):
        payload = _make_payload()
        selected, skipped = _select_cases(
            payload, case_ids=["C001", "C999"], max_cases=None, exclude_types=[]
        )
        self.assertEqual([c["id"] for c in selected], ["C001"])
        self.assertEqual(skipped, ["C999"])

    def test_raises_when_nothing_left_after_filtering(self):
        payload = {"cases": [_make_case("C001", "回归测试")]}
        with self.assertRaises(ValueError):
            _select_cases(payload, case_ids=[], max_cases=None, exclude_types=["回归测试"])


class TestExecuteBatch(unittest.IsolatedAsyncioTestCase):
    async def test_runs_all_selected_cases_without_stopping_on_failure(self):
        driver_manager = MagicMock()
        driver_manager.probe_session_alive = AsyncMock(return_value=True)
        driver_manager.is_connected = MagicMock(return_value=False)

        workflow = _make_workflow(driver_manager)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "automation.json"
            path.write_text(json.dumps(_make_payload()), encoding="utf-8")
            request = ScenarioExecutionRequest(
                automation_json_path=str(path),
                max_cases=2,
                exclude_types=["回归测试"],
            )
            batch_result = await workflow.execute_batch(request)

        # 两条非回归用例都被执行了（虽然都因缺少 automation_steps 而 FAIL），
        # 说明第一条失败没有中断第二条的执行。
        self.assertEqual([r.case_id for r in batch_result.results], ["C001", "C002"])
        self.assertTrue(all(r.status == "FAIL" for r in batch_result.results))
        self.assertEqual(batch_result.pass_count, 0)
        self.assertEqual(batch_result.fail_count, 2)
        self.assertEqual(batch_result.skipped_case_ids, ["C003"])

    async def test_does_not_perform_any_action_tool_calls_between_cases(self):
        # 回归场景：曾经在 case 之间插入过一段"无条件复位滚动"（盲目向上滑动
        # 若干次），后来发现这会把已经展示正常的模块滑走，还容易触发 App
        # 自身的下拉刷新导致状态被意外重置——批量执行不应该在 case 之间
        # 插入任何额外的 action_tool 调用，每条 case 只应该执行它自己
        # automation_steps 里声明的动作。
        driver_manager = MagicMock()
        driver_manager.probe_session_alive = AsyncMock(return_value=True)
        driver_manager.is_connected = MagicMock(return_value=False)

        workflow = _make_workflow(driver_manager)
        workflow._action_tool.execute = AsyncMock()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "automation.json"
            path.write_text(json.dumps(_make_payload()), encoding="utf-8")
            request = ScenarioExecutionRequest(automation_json_path=str(path))
            await workflow.execute_batch(request)

        # 用例本身的 automation_steps 为空，所以整个批量过程中不应该有任何
        # action_tool 调用发生（包括 case 之间）。
        workflow._action_tool.execute.assert_not_awaited()

    async def test_driver_crash_mid_case_fails_only_that_case_not_whole_batch(self):
        # 复现真实报错场景：UiAutomator2 服务端在设备侧偶发崩溃
        # （"instrumentation process is not running"），Appium 对任意指令
        # 都会抛出未捕获的 WebDriverException。如果不在 _execute_step 里
        # 兜底捕获，这个异常会一路冒泡打断 execute_batch 的 for 循环，
        # 导致整个批量执行连同已经跑完的前面几条 case 结果一起丢失。
        # 正确行为：只让"跑到崩溃的那一条 case"判失败，其余 case 正常执行。
        payload = _make_payload()
        payload["cases"][0]["automation_steps"] = [
            {"tool": "action_tool", "action": "tap", "target": "x"}
        ]

        driver_manager = MagicMock()
        driver_manager.probe_session_alive = AsyncMock(return_value=True)
        driver_manager.is_connected = MagicMock(return_value=False)

        workflow = _make_workflow(driver_manager)
        workflow._action_tool.execute = AsyncMock(
            side_effect=RuntimeError(
                "'GET /session/xxx/source' cannot be proxied to UiAutomator2 server "
                "because the instrumentation process is not running (probably crashed)"
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "automation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            request = ScenarioExecutionRequest(automation_json_path=str(path))
            batch_result = await workflow.execute_batch(request)

        # 4 条 case 全部产出了结果（没有因为其中一条驱动崩溃而让整个批次中断）。
        self.assertEqual(
            [r.case_id for r in batch_result.results], ["C001", "C002", "C003", "C004"]
        )
        self.assertEqual(batch_result.results[0].status, "FAIL")
        self.assertIn("UiAutomator2", batch_result.results[0].failure_reason)


if __name__ == "__main__":
    unittest.main()
