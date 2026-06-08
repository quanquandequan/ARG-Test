"""应用层需求服务的检索策略测试。"""

from __future__ import annotations

import json

import pytest

from src.agent.tool_result import ToolExecutionResult
from src.application.artifact_repository import LocalArtifactRepository
from src.application.requirement_context import build_requirement_kb_context
from src.application.requirement_services import (
    RequirementAnalysisService,
    TestCaseGenerationService,
)
from src.application.workflows.test_case_generation_workflow import TestCaseGenerationWorkflow
from src.vectordb.base import SearchResult


class _FakeRetrievalEngine:
    def __init__(self):
        self.calls: list[dict] = []
        self.candidates: list[SearchResult] = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return []

    async def retrieve_candidates(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.candidates)

    async def rerank_candidates(self, query, candidates, top_k=None):
        if top_k is None:
            return list(candidates)
        return list(candidates[:top_k])


class _FakeLoader:
    def load(self, path):
        from src.ingestion.readers.base import Document

        return Document(
            id="upload",
            source_path=str(path),
            content=path.read_text(encoding="utf-8"),
            metadata={},
        )


class _FakeCleaner:
    def clean(self, text: str) -> str:
        return text


class _BrokenAnalyzer:
    async def execute_typed(self, **kwargs):
        return ToolExecutionResult(content="LLM 未能生成有效的 RequirementGraph。")


@pytest.mark.asyncio
async def test_test_case_generation_uses_format_template_query():
    engine = _FakeRetrievalEngine()
    workflow = TestCaseGenerationWorkflow(
        loader=None,
        cleaner=None,
        retrieval_engine=engine,
        artifacts=LocalArtifactRepository(base_dir="./outputs"),
        nodes=[],
    )
    service = TestCaseGenerationService(
        workflow=workflow,
    )

    result = await service._build_kb_samples("追番表Card", "位置在日漫新作上方")

    assert result == ""
    assert engine.calls == [
        {"query": "测试用例 格式 模板", "top_k": 5, "final_k": 3}
    ]


@pytest.mark.asyncio
async def test_requirement_analysis_raises_original_error_when_data_missing(tmp_path):
    service = RequirementAnalysisService(
        loader=_FakeLoader(),
        cleaner=_FakeCleaner(),
        retrieval_engine=_FakeRetrievalEngine(),
        analyzer_tool=_BrokenAnalyzer(),
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
    )

    with pytest.raises(ValueError, match="RequirementGraph"):
        await service.analyze_upload(
            filename="req.txt",
            content="登录需求".encode(),
            module="登录",
        )


@pytest.mark.asyncio
async def test_requirement_kb_context_is_auxiliary_and_excel_first():
    engine = _FakeRetrievalEngine()
    engine.candidates = [
        SearchResult(
            id="bug-1",
            document_id="bugs",
            content="Bug Key: ACNBUG-1 | 旧追番表页面切换异常",
            score=0.95,
            metadata={"source_path": "/kb/ACN_buglist.xlsx", "format": "xlsx"},
        ),
        SearchResult(
            id="excel-1",
            document_id="cases",
            content="动画频道追番表 Card 历史测试用例",
            score=0.80,
            metadata={"source_path": "/kb/ACN_cases.xlsx", "format": "xlsx"},
        ),
    ]

    context = await build_requirement_kb_context(
        engine,
        module="追番表Card",
        requirement_text="新增追番表 Card",
    )

    assert "【历史知识库参考（辅助）】" in context
    assert "不得作为当前需求事实来源" in context
    assert context.index("Excel测试用例") < context.index("Bug记录")


@pytest.mark.asyncio
async def test_test_case_generation_rejects_unconfirmed_analysis_json(tmp_path):
    path = tmp_path / "draft_req_graph.json"
    path.write_text(
        json.dumps({"summary": "草稿", "_meta": {"analysis_status": "draft"}}),
        encoding="utf-8",
    )
    workflow = TestCaseGenerationWorkflow(
        loader=None,
        cleaner=None,
        retrieval_engine=_FakeRetrievalEngine(),
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
        nodes=[],
    )
    service = TestCaseGenerationService(workflow=workflow)

    with pytest.raises(ValueError, match="confirmed"):
        await service.generate_from_analysis_json(str(path))
