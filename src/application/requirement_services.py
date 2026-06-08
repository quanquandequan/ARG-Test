"""需求分析与测试用例生成应用服务。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.agent.tools.requirement_graph_analyzer import RequirementGraphAnalyzerTool
from src.application.artifact_repository import LocalArtifactRepository
from src.application.requirement_context import build_requirement_kb_context
from src.application.workflows import TestCaseGenerationWorkflow
from src.domain.artifacts import ArtifactKind
from src.domain.requirements import (
    RequirementAnalysisResult,
    TestCaseGenerationRequest,
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
        analyzer_tool: RequirementGraphAnalyzerTool,
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
        kb_context = await build_requirement_kb_context(
            self._retrieval_engine,
            resolved_module,
            requirement_text,
        )

        tool_result = await self._analyzer_tool.execute_typed(
            requirement=requirement_text,
            kb_context=kb_context,
            module=resolved_module,
        )
        analysis = tool_result.data
        if analysis is None:
            raise ValueError(tool_result.content or "需求分析未产出有效结构化结果。")

        json_artifact = next(
            (
                artifact
                for artifact in tool_result.artifacts
                if artifact.kind == ArtifactKind.REQUIREMENT_ANALYSIS_JSON
            ),
            None,
        )
        if json_artifact is None:
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
        return RequirementAnalysisResult(
            analysis=analysis,
            json_artifact=json_artifact,
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

class TestCaseGenerationService:
    __test__ = False

    def __init__(
        self,
        workflow: TestCaseGenerationWorkflow,
    ):
        self._workflow = workflow

    async def generate(
        self,
        request: TestCaseGenerationRequest,
        *,
        use_artifact_repository: bool = False,
    ) -> TestCaseGenerationResult:
        generation = await self._workflow.run(request)
        if use_artifact_repository:
            return self._workflow.export_to_artifacts(
                generation,
                output_dir=request.output_dir,
                request_id=request.request_id,
            )
        return self._workflow.export_to_directory(generation, request.output_dir)

    async def generate_from_upload(
        self,
        filename: str,
        content: bytes,
        module: str = "",
        generation_mode: str = "manual",
    ) -> TestCaseGenerationResult:
        requirement_text = self._workflow.load_requirement_text(filename, content)
        resolved_module = module.strip() or "通用"
        kb_samples = await self._build_kb_samples(resolved_module, requirement_text)
        return await self.generate(
            TestCaseGenerationRequest(
                requirement=requirement_text,
                kb_samples=kb_samples,
                module=resolved_module,
                generation_mode=generation_mode,
            ),
            use_artifact_repository=True,
        )

    async def generate_from_text(
        self,
        requirement: str,
        kb_samples: str = "",
        module: str = "",
        output_dir: str = "",
        generation_mode: str = "manual",
        system_prompt_override: str = "",
        request_id: str = "",
        use_artifact_repository: bool = True,
    ) -> TestCaseGenerationResult:
        return await self.generate(
            TestCaseGenerationRequest(
                requirement=requirement,
                kb_samples=kb_samples,
                module=module,
                output_dir=output_dir,
                generation_mode=generation_mode,
                system_prompt_override=system_prompt_override,
                request_id=request_id,
            ),
            use_artifact_repository=use_artifact_repository,
        )

    async def generate_from_analysis_json(
        self,
        analysis_json_path: str,
        module: str = "",
        output_dir: str = "",
        generation_mode: str = "manual",
        system_prompt_override: str = "",
        request_id: str = "",
        use_artifact_repository: bool = True,
    ) -> TestCaseGenerationResult:
        graph = _load_confirmed_analysis_graph(analysis_json_path)
        resolved_module = (
            module.strip()
            or str(graph.get("_meta", {}).get("module") or "")
            or "通用"
        )
        kb_samples = await self._build_kb_samples(
            resolved_module,
            json.dumps(graph, ensure_ascii=False),
        )
        request = TestCaseGenerationRequest(
            requirement=json.dumps(graph, ensure_ascii=False),
            kb_samples=kb_samples,
            module=resolved_module,
            output_dir=output_dir,
            generation_mode=generation_mode,
            system_prompt_override=system_prompt_override,
            request_id=request_id,
        )
        generation = await self._workflow.run_from_analysis_graph(graph, request)
        if use_artifact_repository:
            return self._workflow.export_to_artifacts(
                generation,
                output_dir=request.output_dir,
                request_id=request.request_id,
            )
        return self._workflow.export_to_directory(generation, request.output_dir)

    async def _build_kb_samples(self, module: str, requirement_text: str) -> str:
        return await self._workflow.build_kb_samples(module, requirement_text)

    def _load_requirement_text(self, filename: str, content: bytes) -> str:
        return self._workflow.load_requirement_text(filename, content)


def _load_confirmed_analysis_graph(path: str) -> dict:
    analysis_path = Path(path.strip())
    if not analysis_path.exists():
        raise ValueError(f"确认版需求分析 JSON 不存在：{path}")
    if not analysis_path.is_file():
        raise ValueError(f"确认版需求分析路径不是文件：{path}")
    try:
        graph = json.loads(analysis_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析需求分析 JSON：{exc}") from exc
    if not isinstance(graph, dict):
        raise ValueError("需求分析 JSON 必须是对象。")
    status = str(graph.get("_meta", {}).get("analysis_status") or "")
    if status != "confirmed":
        raise ValueError(
            "请先完成需求确认并生成确认版需求分析 JSON；"
            "当前 JSON 不是 confirmed 状态。"
        )
    return graph
