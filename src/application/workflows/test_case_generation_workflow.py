"""测试用例生成应用工作流。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from src.application.artifact_repository import LocalArtifactRepository
from src.application.exporters import ExcelExporter, JsonExporter, MarkdownExporter
from src.application.exporters.common import normalise_generation_mode
from src.application.workflow_nodes import (
    ArtifactBuilderNode,
    CaseGeneratorNode,
    RequirementParserNode,
    ScenarioGeneratorNode,
    TestPointGeneratorNode,
    WorkflowContext,
    WorkflowNode,
)
from src.core.logging import get_logger
from src.domain.artifacts import ArtifactKind, ArtifactRecord
from src.domain.requirement import Feature, RequirementIR
from src.domain.requirements import (
    TestCaseGenerationData,
    TestCaseGenerationRequest,
    TestCaseGenerationResult,
)
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import SearchResult

logger = get_logger(__name__)


class TestCaseGenerationWorkflow:
    """Application workflow for Requirement -> TestDesignArtifact -> exports."""

    __test__ = False

    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        retrieval_engine: RetrievalEngine,
        artifacts: LocalArtifactRepository,
        nodes: list[WorkflowNode],
        excel_exporter: ExcelExporter | None = None,
        json_exporter: JsonExporter | None = None,
        markdown_exporter: MarkdownExporter | None = None,
        default_output_dir: str = "./outputs/test_cases",
    ):
        self._loader = loader
        self._cleaner = cleaner
        self._retrieval_engine = retrieval_engine
        self._artifacts = artifacts
        self._nodes = nodes
        self._excel_exporter = excel_exporter or ExcelExporter()
        self._json_exporter = json_exporter or JsonExporter()
        self._markdown_exporter = markdown_exporter or MarkdownExporter()
        self._default_output_dir = Path(default_output_dir)

    @classmethod
    def create_default(
        cls,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        retrieval_engine: RetrievalEngine,
        artifacts: LocalArtifactRepository,
        llm,
    ) -> TestCaseGenerationWorkflow:
        return cls(
            loader=loader,
            cleaner=cleaner,
            retrieval_engine=retrieval_engine,
            artifacts=artifacts,
            nodes=[
                RequirementParserNode(llm=llm),
                TestPointGeneratorNode(),
                ScenarioGeneratorNode(),
                CaseGeneratorNode(llm=llm),
                ArtifactBuilderNode(),
            ],
        )

    async def run(self, request: TestCaseGenerationRequest) -> TestCaseGenerationData:
        if not request.requirement or not request.requirement.strip():
            raise ValueError("错误：请提供需求文档内容。")

        module = request.module.strip() or "通用"
        generation_mode = normalise_generation_mode(request.generation_mode)
        context = WorkflowContext(
            request=request,
            requirement_text=request.requirement.strip(),
            module=module,
            generation_mode=generation_mode,
            kb_samples=request.kb_samples,
        )
        for node in self._nodes:
            context = await node.execute(context)

        if context.artifact is None:
            raise ValueError("Workflow did not produce a TestDesignArtifact.")
        return TestCaseGenerationData(
            module=module,
            kb_samples=request.kb_samples,
            generation_mode=generation_mode,
            cases=context.test_cases,
            artifact=context.artifact,
        )

    async def run_from_analysis_graph(
        self,
        graph: dict,
        request: TestCaseGenerationRequest,
    ) -> TestCaseGenerationData:
        """基于确认版 RequirementGraph 生成测试用例，跳过 PRD 二次解析。"""
        module = request.module.strip() or str(
            graph.get("_meta", {}).get("module") or "通用"
        )
        generation_mode = normalise_generation_mode(request.generation_mode)
        requirement_text = _render_analysis_graph_for_generation(graph)
        context = WorkflowContext(
            request=request,
            requirement_text=requirement_text,
            module=module,
            generation_mode=generation_mode,
            kb_samples=request.kb_samples,
            requirement_ir=_requirement_ir_from_analysis_graph(graph, module),
        )
        for node in self._nodes:
            if isinstance(node, RequirementParserNode):
                continue
            context = await node.execute(context)

        if context.artifact is None:
            raise ValueError("Workflow did not produce a TestDesignArtifact.")
        return TestCaseGenerationData(
            module=module,
            kb_samples=request.kb_samples,
            generation_mode=generation_mode,
            cases=context.test_cases,
            artifact=context.artifact,
        )

    def load_requirement_text(self, filename: str, content: bytes) -> str:
        suffix = Path(filename or "upload").suffix
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            doc = self._loader.load(tmp_path)
            return self._cleaner.clean(doc.content).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def build_kb_samples(self, module: str, requirement_text: str) -> str:
        query = f"{module} 测试用例 步骤 预期 格式"
        try:
            candidates = await self._retrieval_engine.retrieve_candidates(
                query=query,
                top_k=80,
            )
        except Exception as exc:
            logger.warning(
                "test_case_kb_samples_unavailable",
                module=module,
                error=str(exc),
            )
            return ""
        excel_candidates = [
            result
            for result in candidates
            if _is_excel_case_sample(result)
        ]
        rerank_pool = excel_candidates or candidates
        try:
            results = await self._retrieval_engine.rerank_candidates(
                query=query,
                candidates=rerank_pool,
                top_k=5,
            )
        except Exception as exc:
            logger.warning(
                "test_case_kb_sample_rerank_unavailable",
                module=module,
                error=str(exc),
            )
            results = rerank_pool[:5]
        results = _select_sample_results(results, final_k=3)
        if not results:
            return ""
        return "\n\n".join(
            f"[{idx}] 来源: {result.document_id}\n{result.content}"
            for idx, result in enumerate(results, start=1)
        )

    def export_to_directory(
        self,
        generation: TestCaseGenerationData,
        output_dir: str = "",
    ) -> TestCaseGenerationResult:
        artifact = _require_artifact(generation)
        out_dir = Path(output_dir.strip()) if output_dir.strip() else self._default_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / _build_filename(generation.module, generation.generation_mode)
        self._excel_exporter.export(artifact, xlsx_path)
        workbook = _record_for_path(ArtifactKind.TEST_CASES_XLSX, xlsx_path)

        automation_json = None
        if generation.generation_mode == "automation":
            json_path = xlsx_path.with_suffix(".json")
            self._json_exporter.export(artifact, json_path)
            automation_json = _record_for_path(
                ArtifactKind.TEST_CASES_AUTOMATION_JSON,
                json_path,
            )

        return TestCaseGenerationResult(
            generation=generation,
            workbook_artifact=workbook,
            automation_json_artifact=automation_json,
            summary=render_generation_summary(generation, workbook, automation_json),
        )

    def export_to_artifacts(
        self,
        generation: TestCaseGenerationData,
        output_dir: str = "",
        request_id: str = "",
    ) -> TestCaseGenerationResult:
        artifact = _require_artifact(generation)
        workbook = self._artifacts.allocate(
            ArtifactKind.TEST_CASES_XLSX,
            generation.module,
            ".xlsx",
            metadata=_artifact_metadata(generation, request_id),
            suffix="automation" if generation.generation_mode == "automation" else "",
            directory=output_dir.strip() or None,
        )
        self._excel_exporter.export(artifact, workbook.path)
        workbook = self._artifacts.finalize(
            workbook,
            metadata={"kb_samples_used": bool(generation.kb_samples.strip())},
        )

        automation_json = None
        if generation.generation_mode == "automation":
            automation_json = self._artifacts.allocate(
                ArtifactKind.TEST_CASES_AUTOMATION_JSON,
                generation.module,
                ".json",
                metadata=_artifact_metadata(generation, request_id),
                suffix="automation",
                directory=output_dir.strip() or None,
            )
            self._json_exporter.export(artifact, automation_json.path)
            automation_json = self._artifacts.finalize(
                automation_json,
                metadata={"kb_samples_used": bool(generation.kb_samples.strip())},
            )

        return TestCaseGenerationResult(
            generation=generation,
            workbook_artifact=workbook,
            automation_json_artifact=automation_json,
            summary=render_generation_summary(generation, workbook, automation_json),
        )


def default_test_case_nodes(llm) -> list[WorkflowNode]:
    return [
        RequirementParserNode(llm=llm),
        TestPointGeneratorNode(),
        ScenarioGeneratorNode(),
        CaseGeneratorNode(llm=llm),
        ArtifactBuilderNode(),
    ]


def render_generation_summary(
    generation: TestCaseGenerationData,
    workbook: ArtifactRecord | None = None,
    automation_json: ArtifactRecord | None = None,
) -> str:
    positive = sum(1 for case in generation.cases if case.case_type in ("正向", "功能"))
    negative = len(generation.cases) - positive
    lines = ["已生成测试用例 Excel 文件："]
    if workbook is not None:
        lines.append(f"路径：{workbook.path}")
    if automation_json is not None:
        lines.append(f"自动化 JSON：{automation_json.path}")
    lines += [
        f"模块：{generation.module}",
        f"生成模式：{generation.generation_mode}",
        f"用例数量：{len(generation.cases)} 条",
        f"（覆盖正向 {positive} 条，反向/边界/异常 {negative} 条）",
    ]
    return "\n".join(lines)


def _require_artifact(generation: TestCaseGenerationData):
    if generation.artifact is None:
        raise ValueError("TestDesignArtifact is required before exporting.")
    return generation.artifact


def _build_filename(module: str, generation_mode: str = "manual") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_module = re.sub(r'[\\/:*?"<>|]', "_", module)
    mode_suffix = "_automation" if generation_mode == "automation" else ""
    return f"{safe_module}{mode_suffix}_{timestamp}.xlsx"


def _record_for_path(kind: ArtifactKind, path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=str(uuid.uuid4()),
        kind=kind,
        path=path.resolve(),
        media_type=_infer_media_type(path.suffix),
    )


def _artifact_metadata(generation: TestCaseGenerationData, request_id: str = "") -> dict:
    metadata = {
        "module": generation.module,
        "case_count": generation.case_count,
        "generation_mode": generation.generation_mode,
    }
    if request_id:
        metadata["request_id"] = request_id
    return metadata


def _requirement_ir_from_analysis_graph(graph: dict, module: str) -> RequirementIR:
    """将确认版 RequirementGraph 转换为测试设计工作流可消费的 RequirementIR。"""
    raw_features = graph.get("features", [])
    features: list[Feature] = []
    for idx, raw in enumerate(raw_features, start=1):
        if not isinstance(raw, dict):
            continue
        priority = str(raw.get("priority", "P1")).strip()
        if priority not in {"P0", "P1", "P2"}:
            priority = "P1"
        boundaries = [str(item) for item in raw.get("boundaries", [])]
        test_focus = [str(item) for item in raw.get("test_focus", [])]
        risk_reason = str(raw.get("risk_reason", "")).strip()
        hints = [*test_focus, *boundaries]
        if risk_reason:
            hints.append(f"风险原因：{risk_reason}")
        features.append(
            Feature(
                id=str(raw.get("id") or f"F{idx:03d}"),
                name=str(raw.get("name") or f"功能点{idx}"),
                description=str(raw.get("description") or ""),
                priority=priority,
                acceptance_criteria=boundaries,
                test_hints=hints,
                dependencies=[str(item) for item in raw.get("dependencies", [])],
            )
        )

    if not features:
        raise ValueError("确认版需求分析 JSON 中没有 features，无法生成测试用例。")

    strategy = graph.get("test_strategy", {})
    exclusions = []
    if isinstance(strategy, dict):
        exclusions = [str(item) for item in strategy.get("exclusions", [])]

    return RequirementIR(
        module=module,
        summary=str(graph.get("summary") or f"{module} 需求分析"),
        features=features,
        out_of_scope=exclusions,
        source_length=len(json.dumps(graph, ensure_ascii=False)),
        has_kb_context=bool(graph.get("_meta", {}).get("has_kb_context")),
    )


def _render_analysis_graph_for_generation(graph: dict) -> str:
    """压缩确认版分析 JSON，作为 CaseGenerator 的唯一需求输入。"""
    payload = {
        "summary": graph.get("summary", ""),
        "features": [
            _compact_feature(feature)
            for feature in graph.get("features", [])
            if isinstance(feature, dict)
        ],
        "state_transitions": _compact_state_transitions(
            graph.get("state_transitions", [])
        ),
        "test_strategy": _compact_test_strategy(graph.get("test_strategy", {})),
    }
    return "确认版需求分析 JSON：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _select_sample_results(
    results: list[SearchResult],
    final_k: int,
) -> list[SearchResult]:
    excel = [
        result
        for result in results
        if _is_excel_case_sample(result)
    ]
    return (excel or results)[:final_k]


def _is_excel_case_sample(result: SearchResult) -> bool:
    metadata = dict(result.metadata or {})
    source_format = str(
        metadata.get("source_format")
        or metadata.get("format")
        or ""
    ).lower()
    source_path = str(metadata.get("source_path") or "").strip()
    source_name = str(metadata.get("source_name") or "").strip()
    source_ext = str(metadata.get("source_ext") or "").lower().strip()
    filename = (source_name or Path(source_path).name or result.document_id).lower()
    ext = source_ext or Path(filename).suffix.lower()
    is_excel = source_format in {"xlsx", "xlsm"} or ext in {".xlsx", ".xlsm"}
    is_bug = any(
        marker in filename
        for marker in ("bug", "buglist", "缺陷", "acn_buglist")
    )
    return is_excel and not is_bug


def _compact_feature(feature: dict) -> dict:
    return {
        "id": feature.get("id", ""),
        "name": feature.get("name", ""),
        "description": feature.get("description", ""),
        "priority": feature.get("priority", "P1"),
        "boundaries": feature.get("boundaries", []),
        "test_focus": feature.get("test_focus", []),
        "dependencies": feature.get("dependencies", []),
    }


def _compact_state_transitions(raw_transitions) -> list[dict]:
    transitions: list[dict] = []
    if not isinstance(raw_transitions, list):
        return transitions
    for item in raw_transitions:
        if not isinstance(item, dict):
            continue
        transitions.append({
            "entity": item.get("entity", ""),
            "states": item.get("states", []),
            "transitions": item.get("transitions", []),
        })
    return transitions


def _compact_test_strategy(strategy) -> dict:
    if not isinstance(strategy, dict):
        return {}
    return {
        "scope": strategy.get("scope", ""),
        "focus_areas": strategy.get("focus_areas", []),
        "exclusions": strategy.get("exclusions", []),
    }


def _infer_media_type(extension: str) -> str:
    mapping = {
        ".json": "application/json",
        ".md": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return mapping.get(extension.lower(), "application/octet-stream")
