"""测试用例生成应用工作流。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from src.core.logging import get_logger
from src.domain.artifacts import ArtifactKind, ArtifactRecord
from src.domain.requirement import Feature, RequirementIR
from src.domain.test_design.generation import (
    TestCaseGenerationData,
    TestCaseGenerationRequest,
    TestCaseGenerationResult,
)
from src.services.case_generator import CaseGeneratorNode
from src.services.exporters import ExcelExporter, JsonExporter, MarkdownExporter
from src.services.exporters.common import normalise_generation_mode
from src.services.workflow_base import WorkflowContext
from src.vectordb.base import SearchResult

logger = get_logger(__name__)


class TestCaseGenerationWorkflow:
    """Application workflow for Requirement -> TestDesignArtifact -> exports."""

    __test__ = False

    def __init__(self, loader, cleaner, retrieval_engine, artifacts, llm=None,
                 excel_exporter=None, json_exporter=None, markdown_exporter=None,
                 default_output_dir="./outputs/test_cases"):
        self._loader = loader
        self._cleaner = cleaner
        self._retrieval_engine = retrieval_engine
        self._artifacts = artifacts
        self._llm = llm
        self._excel_exporter = excel_exporter or ExcelExporter()
        self._json_exporter = json_exporter or JsonExporter()
        self._markdown_exporter = markdown_exporter or MarkdownExporter()
        self._default_output_dir = Path(default_output_dir)

    @classmethod
    def create_default(cls, loader, cleaner, retrieval_engine, artifacts, llm):
        return cls(loader=loader, cleaner=cleaner, retrieval_engine=retrieval_engine, artifacts=artifacts, llm=llm)

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
        # ── Inline pipeline (Step 1-5) ──
        from src.domain.artifacts.test_design_artifact import TestDesignArtifact
        from src.domain.test_design.execution_plan import ExecutionAction, ExecutionPlan
        from src.domain.test_design.test_point import TestPoint
        from src.domain.test_design.test_scenario import TestScenario
        from src.services.requirement_ir_builder import RequirementIRBuilder

        builder = RequirementIRBuilder(llm=self._llm)
        context.requirement_ir = await builder.build(context.requirement_text, context.module, context.kb_samples)
        if context.requirement_ir is None:
            raise ValueError("LLM 未能生成有效的 RequirementIR。")

        ir = context.requirement_ir
        points = []
        for idx, f in enumerate(ir.features, start=1):
            points.append(TestPoint(id=f"TP{idx:03d}", title=f.name, feature_id=f.id, priority=f.priority, test_type="功能", risk_level="medium", source=f"feature:{f.id}", hints=list(f.test_hints)))
        for idx, r in enumerate(ir.business_rules, start=1):
            points.append(TestPoint(id=f"TP{len(points)+1:03d}", title=r.description, priority="P1", test_type="规则", risk_level="medium", source=f"rule:{r.id}", hints=[r.condition, r.outcome]))
        if not points:
            points.append(TestPoint(id="TP001", title=context.module, priority="P0", test_type="功能", source="requirement"))
        context.test_points = points

        scenarios = []
        for idx, pt in enumerate(points, start=1):
            scenarios.append(TestScenario(id=f"SC{idx:03d}", title=pt.title, point_id=pt.id, precondition="无", steps_intent=[f"验证 {pt.title}"], expected_intent=[f"{pt.title} 符合需求"], data_state="normal", priority=pt.priority, test_type=pt.test_type, execution_intent=ExecutionPlan(actions=[ExecutionAction(action="verify", target=pt.title, locator_hints=[pt.title])], assertions=[pt.title], locator_hints=[pt.title])))
        context.scenarios = scenarios

        case_gen = CaseGeneratorNode(llm=self._llm)
        context = await case_gen.execute(context)

        context.artifact = TestDesignArtifact(module=context.module, generation_mode=context.generation_mode, requirement_ir=context.requirement_ir, test_points=context.test_points, scenarios=context.scenarios, test_cases=context.test_cases, metadata={"kb_samples_used": bool(context.kb_samples.strip()), "source_length": len(context.requirement_text)})

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
        # ── Inline pipeline (skip RequirementIR, use graph directly) ──
        from src.domain.artifacts.test_design_artifact import TestDesignArtifact
        from src.domain.test_design.execution_plan import ExecutionAction, ExecutionPlan
        from src.domain.test_design.test_point import TestPoint
        from src.domain.test_design.test_scenario import TestScenario

        features = graph.get("features", [])
        rules = graph.get("business_rules", [])
        points = []
        for idx, f in enumerate(features, start=1):
            points.append(TestPoint(id=f"TP{idx:03d}", title=f.get("name",""), feature_id=f.get("id",""), priority=f.get("priority","P1"), test_type="功能", risk_level=f.get("risk_level","medium"), source=f"feature:{f.get('id','')}", hints=list(f.get("test_focus",[]))))
        for idx, r in enumerate(rules, start=1):
            points.append(TestPoint(id=f"TP{len(points)+1:03d}", title=r.get("description",""), priority="P1", test_type="规则", risk_level="medium", source=f"rule:{r.get('id','')}", hints=[r.get("condition",""), r.get("outcome","")]))
        if not points:
            points.append(TestPoint(id="TP001", title=module, priority="P0", test_type="功能", source="requirement"))
        context.test_points = points

        scenarios = []
        for idx, pt in enumerate(points, start=1):
            scenarios.append(TestScenario(id=f"SC{idx:03d}", title=pt.title, point_id=pt.id, precondition="无", steps_intent=[f"验证 {pt.title}"], expected_intent=[f"{pt.title} 符合需求"], data_state="normal", priority=pt.priority, test_type=pt.test_type, execution_intent=ExecutionPlan(actions=[ExecutionAction(action="verify", target=pt.title, locator_hints=[pt.title])], assertions=[pt.title], locator_hints=[pt.title])))
        context.scenarios = scenarios

        case_gen = CaseGeneratorNode(llm=self._llm)
        context = await case_gen.execute(context)

        context.artifact = TestDesignArtifact(module=context.module, generation_mode=context.generation_mode, requirement_ir=context.requirement_ir, test_points=context.test_points, scenarios=context.scenarios, test_cases=context.test_cases, metadata={"kb_samples_used": bool(context.kb_samples.strip()), "source_length": len(context.requirement_text)})

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

        workbook = None
        automation_json = None
        if generation.generation_mode == "automation":
            json_path = out_dir / _build_json_filename(generation.module)
            self._json_exporter.export(artifact, json_path)
            automation_json = _record_for_path(
                ArtifactKind.TEST_CASES_AUTOMATION_JSON,
                json_path,
            )
        else:
            xlsx_path = out_dir / _build_filename(generation.module, generation.generation_mode)
            self._excel_exporter.export(artifact, xlsx_path)
            workbook = _record_for_path(ArtifactKind.TEST_CASES_XLSX, xlsx_path)

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

        workbook = None
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
        else:
            workbook = self._artifacts.allocate(
                ArtifactKind.TEST_CASES_XLSX,
                generation.module,
                ".xlsx",
                metadata=_artifact_metadata(generation, request_id),
                suffix="",
                directory=output_dir.strip() or None,
            )
            self._excel_exporter.export(artifact, workbook.path)
            workbook = self._artifacts.finalize(
                workbook,
                metadata={"kb_samples_used": bool(generation.kb_samples.strip())},
            )

        return TestCaseGenerationResult(
            generation=generation,
            workbook_artifact=workbook,
            automation_json_artifact=automation_json,
            summary=render_generation_summary(generation, workbook, automation_json),
        )




def render_generation_summary(
    generation: TestCaseGenerationData,
    workbook: ArtifactRecord | None = None,
    automation_json: ArtifactRecord | None = None,
) -> str:
    # 兼容多种 case_type 命名：如“正向”“功能”“功能测试”等都视为正向/主流程类。
    def _is_positive(case_type: str) -> bool:
        text = str(case_type or "").strip()
        return ("正向" in text) or ("功能" in text)

    positive = sum(1 for case in generation.cases if _is_positive(case.case_type))
    negative = len(generation.cases) - positive
    lines = []
    if generation.generation_mode == "automation":
        lines.append("已生成自动化用例定义 JSON：")
        if automation_json is not None:
            lines.append(f"路径：{automation_json.path}")
    else:
        lines.append("已生成测试用例 Excel 文件：")
        if workbook is not None:
            lines.append(f"路径：{workbook.path}")
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


def _build_json_filename(module: str) -> str:
    """automation 模式直接导出 JSON，不再生成中间 Excel。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_module = re.sub(r'[\\/:*?"<>|]', "_", module)
    return f"{safe_module}_automation_{timestamp}.json"


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
