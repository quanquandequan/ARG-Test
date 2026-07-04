"""case_generator 自动化异常用例过滤单元测试。"""

import unittest

from src.services.case_generator import (
    _exclude_exception_cases,
    _is_exception_case_type,
    _render_automation_batch_requirement,
)


class TestIsExceptionCaseType(unittest.TestCase):
    def test_exception_markers_detected(self):
        self.assertTrue(_is_exception_case_type("异常测试"))
        self.assertTrue(_is_exception_case_type("Exception"))

    def test_non_exception_types_pass(self):
        self.assertFalse(_is_exception_case_type("功能测试"))
        self.assertFalse(_is_exception_case_type("UI测试"))
        self.assertFalse(_is_exception_case_type("交互测试"))
        self.assertFalse(_is_exception_case_type("回归测试"))
        self.assertFalse(_is_exception_case_type(""))


class TestExcludeExceptionCases(unittest.TestCase):
    def test_drops_only_exception_cases(self):
        cases = [
            {"title": "验证模块展示", "type": "功能测试"},
            {"title": "验证接口失败静默降级", "type": "异常测试"},
            {"title": "验证点击跳转", "type": "交互测试"},
        ]
        kept = _exclude_exception_cases(cases)
        self.assertEqual([c["title"] for c in kept], ["验证模块展示", "验证点击跳转"])

    def test_empty_list_returns_empty(self):
        self.assertEqual(_exclude_exception_cases([]), [])

    def test_all_exception_returns_empty(self):
        cases = [{"title": "a", "type": "异常测试"}, {"title": "b", "type": "exception"}]
        self.assertEqual(_exclude_exception_cases(cases), [])


class TestRenderAutomationBatchRequirement(unittest.TestCase):
    def test_regression_scope_passthrough(self):
        requirement_text = (
            "确认版需求分析 JSON：\n"
            '{"summary": "追番表卡片", "features": [{"id": "F001", "name": "加追按钮"}], '
            '"state_transitions": [], "test_strategy": {}, '
            '"regression_scope": ["动画频道推荐页现有楼层滑动逻辑"]}'
        )
        rendered = _render_automation_batch_requirement(
            requirement_text, [{"id": "F001", "name": "加追按钮"}]
        )
        self.assertIn("动画频道推荐页现有楼层滑动逻辑", rendered)

    def test_missing_regression_scope_defaults_empty(self):
        requirement_text = (
            "确认版需求分析 JSON：\n"
            '{"summary": "追番表卡片", "features": [], '
            '"state_transitions": [], "test_strategy": {}}'
        )
        rendered = _render_automation_batch_requirement(requirement_text, [])
        self.assertIn('"regression_scope": []', rendered)


if __name__ == "__main__":
    unittest.main()
