"""execute_scenario 硬路由的批量限定词检测单元测试。

回归背景：用户要求"跑前 6 条 case，回归测试的 case 不需要执行"时，硬路由
之前会无视这些限定词、直接只透传 automation_json_path 给 execute_scenario，
导致批量意图完全丢失（只会执行 cases[0]）。现在检测到"前N条""跳过""回归"
等限定词时，硬路由应该让步给 LLM 正常工具调用路径，交由 LLM 结合 route
hint 把约束换算成 max_cases/exclude_types/case_ids 参数。
"""

import unittest
from unittest.mock import MagicMock

from src.agent.react_loop import ReActAgent, _has_batch_qualifier


def _make_agent() -> ReActAgent:
    return ReActAgent(llm=MagicMock(), tools=[], system_prompt="测试用系统提示词")


class TestHasBatchQualifier(unittest.TestCase):
    def test_detects_count_qualifier(self):
        self.assertTrue(_has_batch_qualifier("只跑前6条case，回归测试的case不需要执行"))

    def test_detects_regression_keyword_alone(self):
        self.assertTrue(_has_batch_qualifier("跳过回归用例，其他都执行"))

    def test_plain_execute_request_has_no_qualifier(self):
        self.assertFalse(_has_batch_qualifier("帮我执行一下这个用例"))


class TestExecuteScenarioHardRouteSkipsOnBatchQualifier(unittest.TestCase):
    def setUp(self):
        self.agent = _make_agent()
        self.path = "/tmp/叭嗒动画频道-追番表Card_5027946b_automation.json"

    def test_plain_execute_request_still_hard_routes(self):
        query = f"'{self.path}'帮我执行一下这个case"
        tool_call = self.agent._build_direct_tool_call(query)
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.name, "execute_scenario")
        self.assertEqual(tool_call.arguments, {"automation_json_path": self.path})

    def test_batch_qualifier_defers_to_llm_instead_of_hard_routing(self):
        query = f"'{self.path}'帮我跑一遍case，只跑前6条case，回归测试的case不需要执行"
        tool_call = self.agent._build_direct_tool_call(query)
        self.assertIsNone(tool_call)

    def test_route_hint_still_points_to_execute_scenario_when_deferred(self):
        query = f"'{self.path}'帮我跑一遍case，只跑前6条case，回归测试的case不需要执行"
        hint = self.agent._build_route_hint(query)
        self.assertIn("execute_scenario", hint)
        self.assertIn("max_cases", hint)
        self.assertIn("exclude_types", hint)


if __name__ == "__main__":
    unittest.main()
