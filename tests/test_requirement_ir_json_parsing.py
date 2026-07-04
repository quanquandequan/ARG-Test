"""RequirementIR / ReviewResult / AnalysisReport 的 LLM JSON 解析单元测试。

回归背景：需求文档里经常引用 UI 文案（如"加追""追番表"），LLM 生成 JSON 时
容易把这类中文引号写成和外层字符串相同的 ASCII 双引号，破坏 JSON 结构。
RequirementIR.from_llm_json 曾经缺少这层中文引号修复（ReviewResult 和
AnalysisReport 都有），导致引用了 UI 文案的需求文档解析必现失败。三者现在
共用 _parse_llm_model_json + _repair_llm_json_quotes，这里验证修复对三者都生效。
"""

import unittest

from src.domain.requirement.requirement_ir import (
    AnalysisReport,
    RequirementIR,
    ReviewResult,
)

# 模拟 LLM 输出中把 UI 文案的中文引号误写成和外层相同的 ASCII 双引号
_BROKEN_REQUIREMENT_JSON = """
{
  "module": "追番表Card",
  "summary": "新增追番表 Card 模块",
  "features": [
    {
      "id": "F1",
      "name": "加追按钮",
      "description": "用户点击"加追"按钮后，不拉起登录流程，直接执行追更逻辑",
      "priority": "P1",
      "acceptance_criteria": ["点击"加追"后按钮变为已加追"],
      "test_hints": []
    }
  ],
  "business_rules": [],
  "state_machines": [],
  "data_entities": [],
  "out_of_scope": []
}
"""

_BROKEN_REVIEW_JSON = """
{
  "overall_quality": "needs_clarification",
  "score": 60,
  "ambiguities": [
    {
      "id": "A1",
      "location": "6.6 加追按钮",
      "description": "未说明"加追"按钮在弱网下的重试策略",
      "suggestion": ""
    }
  ],
  "gaps": [],
  "risks": [],
  "suggestions": []
}
"""

_BROKEN_ANALYSIS_JSON = """
{
  "risk_graph": {"nodes": ["F1"], "edges": []},
  "test_strategy": [],
  "clarifications": [],
  "kb_references": ["点击"加追"按钮完成追更"],
  "regression_scope": ["书架页"]
}
"""


class TestRequirementIRFromLlmJson(unittest.TestCase):
    def test_repairs_embedded_chinese_quotes(self):
        ir = RequirementIR.from_llm_json(_BROKEN_REQUIREMENT_JSON)
        self.assertIsNotNone(ir)
        self.assertEqual(ir.module, "追番表Card")
        self.assertIn("「加追」", ir.features[0].description)

    def test_returns_none_for_garbage_input(self):
        self.assertIsNone(RequirementIR.from_llm_json("这不是 JSON"))

    def test_parses_clean_json_directly(self):
        clean = """{"module": "M", "summary": "S", "features": []}"""
        ir = RequirementIR.from_llm_json(clean)
        self.assertIsNotNone(ir)
        self.assertEqual(ir.module, "M")


class TestReviewResultFromLlmJson(unittest.TestCase):
    def test_repairs_embedded_chinese_quotes(self):
        review = ReviewResult.from_llm_json(_BROKEN_REVIEW_JSON)
        self.assertIsNotNone(review)
        self.assertIn("「加追」", review.ambiguities[0].description)


class TestAnalysisReportFromLlmJson(unittest.TestCase):
    def test_repairs_embedded_chinese_quotes(self):
        report = AnalysisReport.from_llm_json(_BROKEN_ANALYSIS_JSON)
        self.assertIsNotNone(report)
        self.assertEqual(report.regression_scope, ["书架页"])
        self.assertIn("「加追」", report.kb_references[0])


if __name__ == "__main__":
    unittest.main()
