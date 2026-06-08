"""高阶需求分析工具：统一解析、评审与测试视角分析。"""

from __future__ import annotations

from pathlib import Path

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.agent.tools.requirement_graph_analyzer import RequirementGraphAnalyzerTool
from src.agent.tools.requirement_parser import RequirementParserTool
from src.agent.tools.requirement_reviewer import RequirementReviewerTool
from src.services.requirement_context import build_requirement_kb_context
from src.services.requirement_input import (
    RequirementInputError,
    resolve_requirement_input,
)
from src.domain.artifacts import ArtifactKind, ArtifactRecord
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.llm.base import BaseLLM
from src.retriever.retrieval_engine import RetrievalEngine


class AnalyzeRequirementTool(BaseTool):
    """对 Agent 暴露单一的需求分析入口。"""

    final_answer_mode = FINAL_ANSWER_PASSTHROUGH

    def __init__(
        self,
        llm: BaseLLM,
        retrieval_engine: RetrievalEngine,
        loader: DocumentLoader | None = None,
        cleaner: TextCleaner | None = None,
    ):
        self._retrieval_engine = retrieval_engine
        self._loader = loader or DocumentLoader()
        self._cleaner = cleaner or TextCleaner()
        self._parser_tool = RequirementParserTool(llm=llm)
        self._reviewer_tool = RequirementReviewerTool(llm=llm)
        self._analyzer_tool = RequirementGraphAnalyzerTool(llm=llm)

    @property
    def name(self) -> str:
        return "analyze_requirement"

    @property
    def description(self) -> str:
        return (
            "对需求文档执行完整测试视角分析：先解析 RequirementIR，"
            "再识别主要歧义、缺口与风险，并生成结构化分析结果。"
            "当前 PRD 是唯一需求事实来源；知识库只作为历史功能和回测范围参考。"
            "默认 draft 模式只返回待确认问题且不生成 JSON；"
            "用户答复后使用 final 模式生成 confirmed JSON。"
            "如果用户提供本地文件路径，可通过 requirement_file 传入。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "requirement": {
                    "type": "string",
                    "description": (
                        "需求文档的完整文本内容；也兼容短指令中的本地文件路径"
                    ),
                },
                "requirement_file": {
                    "type": "string",
                    "description": "本地需求文件路径（可选，支持 txt/md/pdf/xlsx/xmind）",
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名称（可选）",
                },
                "kb_context": {
                    "type": "string",
                    "description": "知识检索工具返回的背景信息（可选）",
                },
                "output_dir": {
                    "type": "string",
                    "description": "统一输出目录（可选）",
                },
                "analysis_mode": {
                    "type": "string",
                    "enum": ["draft", "final"],
                    "description": (
                        "draft=先输出待确认问题且不生成 JSON；"
                        "final=基于用户答复生成确认版 JSON"
                    ),
                },
                "clarification_answers": {
                    "type": "string",
                    "description": "用户针对待确认问题给出的答复；final 模式必填",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        requirement: str = "",
        requirement_file: str = "",
        module: str = "",
        kb_context: str = "",
        output_dir: str = "",
        analysis_mode: str = "draft",
        clarification_answers: str = "",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            requirement=requirement,
            requirement_file=requirement_file,
            module=module,
            kb_context=kb_context,
            output_dir=output_dir,
            analysis_mode=analysis_mode,
            clarification_answers=clarification_answers,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        requirement: str = "",
        requirement_file: str = "",
        module: str = "",
        kb_context: str = "",
        output_dir: str = "",
        request_id: str = "",
        analysis_mode: str = "draft",
        clarification_answers: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        mode = (analysis_mode or "draft").strip().lower()
        if mode not in {"draft", "final"}:
            return ToolExecutionResult(content="错误：analysis_mode 仅支持 draft 或 final。")
        if mode == "final" and not clarification_answers.strip():
            return ToolExecutionResult(
                content="错误：final 模式需要提供 clarification_answers。"
            )

        try:
            requirement_input = resolve_requirement_input(
                requirement=requirement,
                requirement_file=requirement_file,
                loader=self._loader,
                cleaner=self._cleaner,
            )
        except RequirementInputError as exc:
            return ToolExecutionResult(content=f"错误：{exc}")

        resolved_module = module.strip() or "需求分析"
        parser_requirement_text = (
            _merge_requirement_with_answers(
                requirement_input.content,
                clarification_answers,
            )
            if mode == "final"
            else requirement_input.content
        )
        resolved_kb_context = kb_context.strip() or await build_requirement_kb_context(
            self._retrieval_engine,
            resolved_module,
            requirement_input.content,
        )

        parser_result = await self._parser_tool.execute_typed(
            requirement=parser_requirement_text,
            kb_context="",
            module=resolved_module,
            output_dir=output_dir,
            request_id=request_id,
            persist=False,
        )
        if parser_result.data is None:
            return _failed_result(
                "requirement_parser",
                parser_result,
                request_id=request_id,
            )

        reviewer_result = await self._reviewer_tool.execute_typed(
            ir_file="",
            ir_json=parser_result.data.model_dump_json(),
            requirement="",
            module=resolved_module,
            output_dir=output_dir,
            request_id=request_id,
            persist=False,
        )
        if reviewer_result.data is None:
            return _failed_result(
                "requirement_reviewer",
                reviewer_result,
                request_id=request_id,
                artifacts=[*parser_result.artifacts, *reviewer_result.artifacts],
            )

        analyzer_result = await self._analyzer_tool.execute_typed(
            requirement=requirement_input.content,
            kb_context=resolved_kb_context,
            module=resolved_module,
            output_dir=output_dir,
            request_id=request_id,
            persist=(mode == "final"),
            analysis_status="confirmed" if mode == "final" else "draft",
            requirement_source_path=requirement_input.source_path,
            clarification_answers=clarification_answers,
        )
        if analyzer_result.data is None:
            return _failed_result(
                "analyze_requirements",
                analyzer_result,
                request_id=request_id,
                artifacts=[
                    *parser_result.artifacts,
                    *reviewer_result.artifacts,
                    *analyzer_result.artifacts,
                ],
            )

        if mode == "draft":
            return _draft_result(
                module=resolved_module,
                requirement_source=requirement_input.source_path or "当前输入的需求文本",
                parser_result=parser_result,
                reviewer_result=reviewer_result,
                analyzer_result=analyzer_result,
                request_id=request_id,
                has_kb_context=bool(resolved_kb_context.strip()),
            )

        artifacts = list(analyzer_result.artifacts)

        review = reviewer_result.data
        analysis = analyzer_result.data
        analysis_json = _first_artifact(artifacts, ArtifactKind.REQUIREMENT_ANALYSIS_JSON)

        lines = [
            f"确认版需求分析完成：{resolved_module}",
            f"需求事实源：{requirement_input.source_path or '当前输入的需求文本'}",
            "KB 用途：仅作为历史功能、历史差异和回测范围参考，不作为当前需求事实。",
            "确认补充：已合并用户针对待确认问题的答复。",
            "",
        ]
        if analysis_json is not None:
            lines.append(f"分析结果：{analysis_json.path}")
        lines += [
            "",
            f"摘要：{parser_result.data.summary}",
        ]
        if parser_result.data.features:
            lines += ["", "当前 PRD 功能点："]
            for feature in parser_result.data.features[:5]:
                lines.append(f"  [{feature.id}] {feature.name}")

        if review is not None:
            lines.append(f"质量评分：{review.score}/100（{review.overall_quality}）")
            if review.ambiguities:
                lines += ["", "主要歧义："]
                for ambiguity in review.ambiguities[:3]:
                    lines.append(f"  [{ambiguity.id}] {ambiguity.description}")
            if review.gaps:
                lines += ["", "主要缺口："]
                for gap in review.gaps[:3]:
                    lines.append(f"  [{gap.id}] {gap.question}")
            high_risks = [risk for risk in review.risks if risk.level == "high"]
            if high_risks:
                lines += ["", "高风险："]
                for risk in high_risks[:3]:
                    lines.append(f"  [{risk.area}] {risk.description}")

        if analysis is not None:
            lines += [
                "",
                f"功能点：{analysis.feature_count} 个",
                f"风险点：{analysis.risk_count} 个",
                f"待澄清问题：{analysis.clarification_count} 个",
            ]
            _append_analysis_reference_summary(lines, analysis.graph)

        return ToolExecutionResult(
            content="\n".join(lines),
            data={
                "ir": parser_result.data,
                "review": review,
                "analysis": analysis,
            },
            artifacts=artifacts,
            metadata={
                "request_id": request_id,
                "requirement_source_path": requirement_input.source_path,
                "has_kb_context": bool(resolved_kb_context.strip()),
                "analysis_status": "confirmed",
                "output_dir": str(
                    (Path(output_dir) if output_dir.strip() else Path.cwd()).resolve()
                ),
            },
        )


def _first_artifact(
    artifacts: list[ArtifactRecord],
    kind: ArtifactKind,
) -> ArtifactRecord | None:
    for artifact in artifacts:
        if artifact.kind == kind:
            return artifact
    return None


def _failed_result(
    stage: str,
    result: ToolExecutionResult,
    *,
    request_id: str,
    artifacts: list[ArtifactRecord] | None = None,
) -> ToolExecutionResult:
    """将子工具结构化失败转换为复合工具的明确失败。"""
    return ToolExecutionResult(
        content=(
            f"需求分析失败：{stage} 未产出有效结构化结果。\n"
            f"{result.content}"
        ),
        artifacts=artifacts if artifacts is not None else list(result.artifacts),
        metadata={
            "request_id": request_id,
            "failed_stage": stage,
        },
    )


def _merge_requirement_with_answers(
    requirement: str,
    clarification_answers: str,
) -> str:
    """将用户确认答复作为本次需求事实补充附加到 PRD 后。"""
    if not clarification_answers.strip():
        return requirement
    return (
        f"{requirement.strip()}\n\n"
        "【用户确认补充】\n"
        "以下内容是用户针对待确认问题给出的答复，属于本次需求事实补充：\n"
        f"{clarification_answers.strip()}"
    )


def _draft_result(
    *,
    module: str,
    requirement_source: str,
    parser_result: ToolExecutionResult,
    reviewer_result: ToolExecutionResult,
    analyzer_result: ToolExecutionResult,
    request_id: str,
    has_kb_context: bool,
) -> ToolExecutionResult:
    """渲染不落盘的需求分析草稿，并提示用户先确认问题。"""
    review = reviewer_result.data
    analysis = analyzer_result.data
    lines = [
        f"需求分析草稿（待确认）：{module}",
        f"需求事实源：{requirement_source}",
        "产物状态：草稿阶段未生成最终 JSON，不可直接用于生成测试用例。",
        "下一步：请逐条回答下方“需求确认问题”；收到答复后我会生成确认版需求分析 JSON。",
        "",
        f"摘要：{parser_result.data.summary}",
    ]

    if parser_result.data.features:
        lines += ["", "当前 PRD 功能点草稿："]
        for feature in parser_result.data.features[:7]:
            lines.append(f"  [{feature.id}] {feature.name}")

    if review is not None:
        lines += ["", f"质量评分：{review.score}/100（{review.overall_quality}）"]
        _append_confirmation_questions(lines, review, analysis)

    if analysis is not None:
        lines += [
            "",
            f"草稿功能点：{analysis.feature_count} 个",
            f"草稿风险点：{analysis.risk_count} 个",
            f"待澄清问题：{analysis.clarification_count} 个",
        ]
        _append_analysis_reference_summary(lines, analysis.graph)

    return ToolExecutionResult(
        content="\n".join(lines),
        data={
            "ir": parser_result.data,
            "review": review,
            "analysis": analysis,
        },
        metadata={
            "request_id": request_id,
            "analysis_status": "draft",
            "persisted": False,
            "has_kb_context": has_kb_context,
            "requirement_source_path": (
                requirement_source if requirement_source != "当前输入的需求文本" else ""
            ),
        },
    )


def _append_confirmation_questions(lines: list[str], review, analysis) -> None:
    """将歧义、缺口和分析器澄清项合并为用户可直接回答的问题清单。"""
    questions: list[tuple[str, str]] = []
    for ambiguity in review.ambiguities[:5]:
        questions.append((
            ambiguity.id,
            f"{ambiguity.description} 请确认具体规则。",
        ))
    for gap in review.gaps[:5]:
        questions.append((gap.id, gap.question))

    graph = analysis.graph if analysis is not None else {}
    clarifications = graph.get("clarifications", []) if isinstance(graph, dict) else []
    existing_ids = {item[0] for item in questions}
    for item in clarifications:
        question_id = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        if not question or question_id in existing_ids:
            continue
        questions.append((question_id or "Q", question))
        existing_ids.add(question_id)
        if len(questions) >= 8:
            break

    if not questions:
        lines += [
            "",
            "需求确认问题：",
            "  无必须确认的问题；如你确认草稿内容无误，可以回复“确认”。",
        ]
        return

    lines += [
        "",
        "需求确认问题：",
        "请按编号逐条回答，回答完成后我会基于你的确认生成 confirmed 需求分析 JSON。",
    ]
    for idx, (question_id, question) in enumerate(questions, start=1):
        prefix = f"  {idx}. "
        source = f"（来源：{question_id}）" if question_id else ""
        lines.append(f"{prefix}{question}{source}")


def _append_analysis_reference_summary(lines: list[str], graph: dict) -> None:
    """从 RequirementGraph 中提取 KB 辅助参考，避免 Agent 另行拼接旧知识。"""
    strategy = graph.get("test_strategy", {}) if isinstance(graph, dict) else {}
    focus_areas = strategy.get("focus_areas", []) if isinstance(strategy, dict) else []
    if focus_areas:
        lines += ["", "回测范围参考："]
        for item in focus_areas[:5]:
            lines.append(f"  - {item}")

    risks = graph.get("risks", []) if isinstance(graph, dict) else []
    if risks:
        lines += ["", "历史/回归风险参考："]
        for risk in risks[:3]:
            area = risk.get("area", "风险")
            desc = risk.get("description", "")
            lines.append(f"  [{area}] {desc}")
