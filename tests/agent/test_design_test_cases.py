"""DesignTestCasesTool 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH
from src.agent.tools.design_test_cases import DesignTestCasesTool
from src.domain.artifacts import ArtifactKind, ArtifactRecord
from src.domain.requirements import (
    GeneratedTestCase,
)
from src.domain.requirements import (
    TestCaseGenerationData as _TestCaseGenerationData,
)
from src.domain.requirements import (
    TestCaseGenerationResult as _TestCaseGenerationResult,
)


class _FakeService:
    def __init__(self):
        self.calls: list[dict] = []

    async def generate_from_analysis_json(self, **kwargs):
        self.calls.append(kwargs)
        case = GeneratedTestCase(
            title="展示追番表Card",
            module=kwargs.get("module") or "追番表Card",
            precondition="存在已确认需求分析 JSON",
            steps="1. 打开动画频道推荐页",
            expected="展示追番表Card",
            priority="P0",
            case_type="正向",
        )
        generation = _TestCaseGenerationData(
            module=kwargs.get("module") or "追番表Card",
            cases=[case],
            generation_mode=kwargs.get("generation_mode") or "manual",
        )
        artifact = ArtifactRecord(
            artifact_id="xlsx-1",
            kind=ArtifactKind.TEST_CASES_XLSX,
            path=Path("/tmp/cases.xlsx"),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        return _TestCaseGenerationResult(
            generation=generation,
            workbook_artifact=artifact,
            summary="已生成测试用例 Excel 文件：\n路径：/tmp/cases.xlsx\n用例数量：1 条",
        )


def test_design_test_cases_requires_analysis_json_path():
    tool = DesignTestCasesTool(service=_FakeService())
    schema = tool.parameters

    assert tool.final_answer_mode == FINAL_ANSWER_PASSTHROUGH
    assert schema["required"] == ["analysis_json_path"]
    assert "analysis_json_path" in schema["properties"]
    assert "requirement" not in schema["properties"]


@pytest.mark.asyncio
async def test_design_test_cases_missing_analysis_json_returns_error():
    service = _FakeService()
    tool = DesignTestCasesTool(service=service)

    result = await tool.execute_typed(
        requirement="需求文件路径：/tmp/req.md",
        module="追番表Card",
    )

    assert "请先完成需求确认" in result.content
    assert service.calls == []


@pytest.mark.asyncio
async def test_design_test_cases_uses_confirmed_analysis_json_path():
    service = _FakeService()
    tool = DesignTestCasesTool(service=service)

    result = await tool.execute_typed(
        analysis_json_path="/tmp/confirmed_req_graph.json",
        module="追番表Card",
        generation_mode="manual",
        request_id="req-1",
    )

    assert "已生成测试用例 Excel 文件" in result.content
    assert service.calls[0]["analysis_json_path"] == "/tmp/confirmed_req_graph.json"
    assert result.metadata["analysis_json_path"] == "/tmp/confirmed_req_graph.json"
