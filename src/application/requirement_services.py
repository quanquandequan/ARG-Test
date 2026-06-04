"""Application services for requirements analysis and test-case generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.agent.tools.analyze_requirements import AnalyzeRequirementsTool
from src.agent.tools.write_test_cases import WriteTestCasesTool
from src.application.artifact_repository import LocalArtifactRepository
from src.domain.artifacts import ArtifactKind
from src.domain.requirements import (
    RequirementAnalysisResult,
    TestCaseGenerationResult,
)
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.retriever.retrieval_engine import RetrievalEngine


class RequirementAnalysisService:
    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        retrieval_engine: RetrievalEngine,
        analyzer_tool: AnalyzeRequirementsTool,
        artifacts: LocalArtifactRepository,
    ):
        self._loader = loader
        self._cleaner = cleaner
        self._retrieval_engine = retrieval_engine
        self._analyzer_tool = analyzer_tool
        self._artifacts = artifacts

    async def analyze_upload(
        self,
        filename: str,
        content: bytes,
        module: str = "",
    ) -> RequirementAnalysisResult:
        requirement_text = self._load_requirement_text(filename, content)
        resolved_module = module.strip() or "需求分析"
        kb_context = await self._build_kb_context(resolved_module, requirement_text)

        tool_result = await self._analyzer_tool.execute_typed(
            requirement=requirement_text,
            kb_context=kb_context,
            module=resolved_module,
        )
        analysis = tool_result.data

        metadata = {
            "module": analysis.module,
            "summary": analysis.summary,
            "feature_count": analysis.feature_count,
            "risk_count": analysis.risk_count,
            "clarification_count": analysis.clarification_count,
        }
        json_artifact = self._artifacts.save_json(
            ArtifactKind.REQUIREMENT_ANALYSIS_JSON,
            analysis.module,
            analysis.graph,
            metadata=metadata,
            suffix="req_graph",
        )
        markdown_artifact = self._artifacts.save_text(
            ArtifactKind.REQUIREMENT_ANALYSIS_MARKDOWN,
            analysis.module,
            self._analyzer_tool.render_markdown(analysis.graph, analysis.module),
            ".md",
            metadata=metadata,
            suffix="analysis",
        )
        return RequirementAnalysisResult(
            analysis=analysis,
            json_artifact=json_artifact,
            markdown_artifact=markdown_artifact,
        )

    def _load_requirement_text(self, filename: str, content: bytes) -> str:
        suffix = Path(filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            doc = self._loader.load(tmp_path)
            return self._cleaner.clean(doc.content).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _build_kb_context(self, module: str, requirement_text: str) -> str:
        query = f"{module} 测试用例".strip() if module.strip() else requirement_text[:60]
        results = await self._retrieval_engine.search(query=query, top_k=5, final_k=5)
        if not results:
            return ""

        lines: list[str] = []
        for idx, result in enumerate(results, start=1):
            lines.append(f"[{idx}] 来源: {result.document_id}\n{result.content}")
        return "\n\n".join(lines)


class TestCaseGenerationService:
    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        retrieval_engine: RetrievalEngine,
        writer_tool: WriteTestCasesTool,
        artifacts: LocalArtifactRepository,
    ):
        self._loader = loader
        self._cleaner = cleaner
        self._retrieval_engine = retrieval_engine
        self._writer_tool = writer_tool
        self._artifacts = artifacts

    async def generate_from_upload(
        self,
        filename: str,
        content: bytes,
        module: str = "",
    ) -> TestCaseGenerationResult:
        requirement_text = self._load_requirement_text(filename, content)
        resolved_module = module.strip() or "通用"
        kb_samples = await self._build_kb_samples(resolved_module, requirement_text)
        tool_result = await self._writer_tool.execute_typed(
            requirement=requirement_text,
            kb_samples=kb_samples,
            module=resolved_module,
        )
        generation = tool_result.data

        artifact = self._artifacts.allocate(
            ArtifactKind.TEST_CASES_XLSX,
            generation.module,
            ".xlsx",
            metadata={"module": generation.module, "case_count": generation.case_count},
        )
        self._writer_tool.write_cases_to_excel(
            generation.cases,
            artifact.path,
            generation.module,
        )
        artifact = self._artifacts.finalize(
            artifact,
            metadata={"kb_samples_used": bool(generation.kb_samples.strip())},
        )

        return TestCaseGenerationResult(
            generation=generation,
            workbook_artifact=artifact,
        )

    def _load_requirement_text(self, filename: str, content: bytes) -> str:
        suffix = Path(filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            doc = self._loader.load(tmp_path)
            return self._cleaner.clean(doc.content).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _build_kb_samples(self, module: str, requirement_text: str) -> str:
        query = f"{module} 测试用例".strip() if module.strip() else requirement_text[:60]
        results = await self._retrieval_engine.search(query=query, top_k=5, final_k=3)
        if not results:
            return ""

        lines: list[str] = []
        for idx, result in enumerate(results, start=1):
            lines.append(f"[{idx}] 来源: {result.document_id}\n{result.content}")
        return "\n\n".join(lines)

