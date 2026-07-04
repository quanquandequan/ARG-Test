"""TestCaseGenerationWorkflow 新增功能影响面回归逻辑单元测试。"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.core.config import load_config
from src.domain.test_design.generated_test_case import GeneratedTestCase
from src.domain.test_design.generation import TestCaseGenerationData
from src.vectordb.base import SearchResult
from src.workflows.testcase_design import (
    TestCaseGenerationWorkflow,
    _render_analysis_graph_for_generation,
    render_generation_summary,
)

load_config()


def _make_excel_result(content: str, rid: str) -> SearchResult:
    return SearchResult(
        id=rid,
        document_id=rid,
        content=content,
        score=0.8,
        metadata={"source_format": "xlsx", "source_name": f"{rid}.xlsx"},
    )


def _make_workflow(retrieval_engine) -> TestCaseGenerationWorkflow:
    return TestCaseGenerationWorkflow(
        loader=MagicMock(),
        cleaner=MagicMock(),
        retrieval_engine=retrieval_engine,
        artifacts=MagicMock(),
        llm=None,
    )


class TestRenderAnalysisGraphForGeneration(unittest.TestCase):
    def test_regression_scope_included_in_payload(self):
        graph = {
            "summary": "追番表卡片",
            "features": [],
            "state_transitions": [],
            "test_strategy": {},
            "regression_scope": ["动画频道推荐页现有楼层滑动逻辑", "  ", ""],
        }
        rendered = _render_analysis_graph_for_generation(graph)
        payload = json.loads(rendered.split("确认版需求分析 JSON：\n", 1)[1])
        # 空白/空字符串条目应被过滤掉，只保留有效描述
        self.assertEqual(payload["regression_scope"], ["动画频道推荐页现有楼层滑动逻辑"])

    def test_empty_regression_scope_defaults_to_empty_list(self):
        graph = {"summary": "x", "features": [], "state_transitions": [], "test_strategy": {}}
        rendered = _render_analysis_graph_for_generation(graph)
        payload = json.loads(rendered.split("确认版需求分析 JSON：\n", 1)[1])
        self.assertEqual(payload["regression_scope"], [])


class TestBuildRegressionContext(unittest.IsolatedAsyncioTestCase):
    async def test_empty_scope_returns_empty_string(self):
        workflow = _make_workflow(retrieval_engine=MagicMock())
        result = await workflow.build_regression_context("动画频道", [])
        self.assertEqual(result, "")

    async def test_queries_kb_per_scope_and_formats_sections(self):
        engine = MagicMock()
        engine.retrieve_candidates = AsyncMock(
            return_value=[_make_excel_result("楼层滑动测试用例原文", "r1")]
        )
        engine.rerank_candidates = AsyncMock(
            return_value=[_make_excel_result("楼层滑动测试用例原文", "r1")]
        )
        workflow = _make_workflow(retrieval_engine=engine)

        result = await workflow.build_regression_context(
            "动画频道", ["动画频道推荐页现有楼层滑动逻辑"]
        )

        self.assertIn("动画频道推荐页现有楼层滑动逻辑", result)
        self.assertIn("楼层滑动测试用例原文", result)
        engine.retrieve_candidates.assert_awaited_once()
        engine.rerank_candidates.assert_awaited_once()

    async def test_retrieval_failure_is_skipped_not_raised(self):
        engine = MagicMock()
        engine.retrieve_candidates = AsyncMock(side_effect=RuntimeError("boom"))
        workflow = _make_workflow(retrieval_engine=engine)

        result = await workflow.build_regression_context("动画频道", ["某个受影响范围"])

        self.assertEqual(result, "")


def _make_case(case_type: str) -> GeneratedTestCase:
    return GeneratedTestCase(
        title="用例",
        module="模块",
        precondition="",
        steps="1、操作",
        expected="符合预期",
        priority="P1",
        case_type=case_type,
    )


class TestRenderGenerationSummary(unittest.TestCase):
    def test_new_taxonomy_case_types_are_not_misclassified_as_exception(self):
        # 新方法论下的 case_type 取值：交互测试/功能测试/UI测试/回归测试都不是异常场景，
        # 只有"异常测试"才应该被计入异常/边界场景。
        generation = TestCaseGenerationData(
            module="模块",
            cases=[
                _make_case("交互测试"),
                _make_case("功能测试"),
                _make_case("UI测试"),
                _make_case("回归测试"),
                _make_case("异常测试"),
            ],
        )
        summary = render_generation_summary(generation)
        self.assertIn("核心场景 4 条", summary)
        self.assertIn("异常/边界场景 1 条", summary)

    def test_all_core_cases_report_zero_exception(self):
        generation = TestCaseGenerationData(
            module="模块",
            cases=[_make_case("交互测试"), _make_case("功能测试")],
        )
        summary = render_generation_summary(generation)
        self.assertIn("核心场景 2 条", summary)
        self.assertIn("异常/边界场景 0 条", summary)


if __name__ == "__main__":
    unittest.main()
