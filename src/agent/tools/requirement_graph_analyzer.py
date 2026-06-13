"""RequirementGraph 分析器：基于 EnrichedIR 生成增量分析报告。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.core.logging import get_logger
from src.core.prompt_loader import require_prompt_fields
from src.domain.artifacts import ArtifactKind
from src.domain.requirement import AnalysisReport, EnrichedRequirementIR
from src.domain.requirement.analysis import RequirementAnalysisData
from src.llm.base import BaseLLM
from src.llm.types import Message
from src.services.artifact_repository import LocalArtifactRepository

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirements"


class RequirementGraphAnalyzerTool(BaseTool):
    """基于 EnrichedRequirementIR 推导风险关系、澄清优先级和测试策略。"""

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str | None = None,
        output_dir: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        from src.core.config import get_config

        cfg = get_config().get("req_analyzer", {})
        self._llm = llm
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self._temperature = (
            temperature if temperature is not None
            else float(cfg.get("temperature", 0.3))
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None
            else int(cfg.get("max_tokens", 8192))
        )
        prompt = require_prompt_fields(
            "requirement_graph_analyzer",
            ["system_prompt", "user_template", "kb_section_template"],
        )
        self._system_prompt = system_prompt or prompt["system_prompt"]
        self._user_template = prompt["user_template"]
        self._kb_section_template = prompt["kb_section_template"]
        self._artifacts = LocalArtifactRepository(base_dir=str(self._default_output_dir))

    @property
    def name(self) -> str:
        return "analyze_requirements"

    @property
    def description(self) -> str:
        return (
            "基于 EnrichedRequirementIR 进行测试视角增量分析，生成结构化 "
            "RequirementGraph 兼容 JSON。输入必须是 parser 与 reviewer "
            "已经产出的 EnrichedRequirementIR；本工具不接收原始 PRD。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "enriched_ir_json": {
                    "type": "string",
                    "description": (
                        "EnrichedRequirementIR JSON 字符串，包含 RequirementIR "
                        "和 ReviewResult。"
                    ),
                },
                "kb_context": {
                    "type": "string",
                    "description": (
                        "知识库背景信息，仅用于识别历史差异、回归风险和回测范围。"
                    ),
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名称，用于文件命名（可选）。",
                },
                "output_dir": {
                    "type": "string",
                    "description": f"输出目录（可选，默认 {_DEFAULT_OUTPUT_DIR}）。",
                },
            },
            "required": ["enriched_ir_json"],
        }

    async def execute(
        self,
        enriched_ir_json: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            enriched_ir_json=enriched_ir_json,
            kb_context=kb_context,
            module=module,
            output_dir=output_dir,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        enriched_ir_json: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        request_id: str = "",
        persist: bool = True,
        analysis_status: str = "",
        requirement_source_path: str = "",
        clarification_answers: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        if not enriched_ir_json.strip():
            return ToolExecutionResult(
                content="错误：请提供 enriched_ir_json。"
            )

        enriched_ir = _parse_enriched_ir(enriched_ir_json)
        if enriched_ir is None:
            return ToolExecutionResult(
                content="错误：enriched_ir_json 不是有效的 EnrichedRequirementIR。"
            )

        resolved_module = module.strip() or enriched_ir.ir.module or "需求分析"
        kb_section = (
            self._kb_section_template.format(context=kb_context.strip())
            if kb_context.strip()
            else ""
        )
        user_content = self._user_template.format(
            kb_section=kb_section,
            enriched_ir_json=enriched_ir.to_graph_analyzer_json(),
            module=resolved_module,
        )
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]

        response = await self._llm.generate_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw = response.content.strip()
        # 内联解析：跳过 from_llm_json，直接多策略尝试
        import json as _json
        import re as _re
        report = None
        
        # 策略1: 去掉 markdown fence 后解析
        cleaned = _re.sub(r"```(?:json)?\s*", "", raw, count=1)
        cleaned = _re.sub(r"\s*```\s*$", "", cleaned, count=1).strip()
        for candidate in [cleaned, _re.search(r"\{[\s\S]*\}", cleaned)]:
            if not candidate:
                continue
            if hasattr(candidate, 'group'):
                candidate = candidate.group()
            try:
                data = _json.loads(candidate)
                if isinstance(data, dict):
                    report = AnalysisReport.model_validate(data)
                    report.graph = data
                    break
            except Exception:
                pass
        
        if report is None:
            # Last-resort debug: try to get the actual parsing error
            err_detail = "unknown"
            try:
                cleaned = _re.sub(r"```(?:json)?\s*", "", raw, count=1)
                cleaned = _re.sub(r"\s*```\s*$", "", cleaned, count=1).strip()
                if cleaned:
                    data = _json.loads(cleaned)
                    AnalysisReport.model_validate(data)
                    err_detail = "JSON valid AND model valid — should have succeeded!"
            except _json.JSONDecodeError as _je:
                err_detail = f"JSON error at pos {_je.pos}: {str(_je)[:150]}"
            except Exception as _ve:
                err_detail = f"Pydantic: {type(_ve).__name__}: {str(_ve)[:200]}"
            logger.warning(
                "analysis_report_parse_failed",
                module=resolved_module,
                raw_len=len(raw),
                raw_first=raw[:100],
                raw_last=raw[-100:],
                debug_err=err_detail,
            )
            return ToolExecutionResult(
                content="LLM 未能生成有效的 AnalysisReport，请重试。"
            )

        validation_errors = _validate_analysis_report(
            report,
            enriched_ir=enriched_ir,
            kb_context=kb_context,
        )
        if validation_errors:
            logger.warning(
                "analysis_report_validation_failed",
                module=resolved_module,
                error_count=len(validation_errors),
            )
            return ToolExecutionResult(
                content=_render_analysis_validation_failure(validation_errors),
                metadata={
                    "request_id": request_id,
                    "validation_error_count": len(validation_errors),
                },
            )

        graph = _build_graph_from_enriched_ir(
            enriched_ir=enriched_ir,
            report=report,
            module=resolved_module,
        )
        graph["_meta"] = {
            "module": resolved_module,
            "generated_at": datetime.now().isoformat(),
            "source_length": enriched_ir.ir.source_length,
            "has_kb_context": bool(kb_context.strip()),
            "analysis_status": analysis_status.strip() or (
                "confirmed" if persist else "draft"
            ),
            "clarification_answers_used": bool(clarification_answers.strip()),
            "evidence_validated": False,
            "analysis_contract": "enriched_ir",
        }
        if requirement_source_path.strip():
            graph["_meta"]["requirement_source_path"] = requirement_source_path.strip()
        if clarification_answers.strip():
            graph["_meta"]["clarification_answers"] = clarification_answers.strip()
        if persist:
            graph["_meta"]["confirmed_at"] = datetime.now().isoformat()

        features = graph.get("features", [])
        risks = graph.get("risks", [])
        clarifications = graph.get("clarifications", [])
        logger.info(
            "analyze_requirements_done",
            module=resolved_module,
            features=len(features),
            risks=len(risks),
            clarifications=len(clarifications),
            analysis_contract="enriched_ir",
        )
        data = RequirementAnalysisData(
            module=resolved_module,
            summary=graph.get("summary", "—"),
            graph=graph,
            feature_count=len(features),
            risk_count=len(risks),
            clarification_count=len(clarifications),
            kb_context=kb_context,
        )
        if not persist:
            return ToolExecutionResult(
                content=_render_analysis_summary(data, json_path=""),
                data=data,
                metadata={
                    "request_id": request_id,
                    "analysis_status": "draft",
                    "persisted": False,
                },
            )

        out_dir = (
            Path(output_dir.strip())
            if output_dir.strip()
            else self._default_output_dir
        )
        metadata = _build_request_metadata(request_id)
        json_artifact = self._artifacts.save_json(
            ArtifactKind.REQUIREMENT_ANALYSIS_JSON,
            resolved_module,
            graph,
            metadata=metadata,
            suffix="req_graph",
            directory=out_dir,
        )
        return ToolExecutionResult(
            content=_render_analysis_summary(data, json_path=str(json_artifact.path)),
            data=data,
            artifacts=[json_artifact],
            metadata={
                "request_id": request_id,
                "output_dir": str(out_dir.resolve()),
            },
        )


def _find_first_json_object(text: str) -> str:
    """从文本中提取第一个顶层 JSON 对象。"""
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group() if match else ""


def _parse_enriched_ir(raw: str) -> EnrichedRequirementIR | None:
    """解析 EnrichedRequirementIR JSON；失败时返回 None。"""
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

    candidates = [text]
    extracted = _find_first_json_object(text)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return EnrichedRequirementIR.model_validate_json(candidate)
        except Exception:
            pass
    return None


def _validate_analysis_report(
    report: AnalysisReport,
    *,
    enriched_ir: EnrichedRequirementIR,
    kb_context: str,
) -> list[str]:
    """校验 AnalysisReport 是否只引用上游已存在的 id 和 KB 文本。"""
    errors: list[str] = []
    feature_ids = {feature.id for feature in enriched_ir.ir.features}
    gap_ids = {gap.id for gap in enriched_ir.review.gaps}
    ambiguity_ids = {a.id for a in enriched_ir.review.ambiguities}
    valid_related_ids = feature_ids | gap_ids | ambiguity_ids

    for node in report.risk_graph.nodes:
        if node not in feature_ids:
            errors.append(f"risk_graph.nodes 包含未知 feature_id：{node}")

    for idx, edge in enumerate(report.risk_graph.edges):
        if edge.from_feature_id not in feature_ids:
            errors.append(
                f"risk_graph.edges[{idx}].from_feature_id 未在 IR 中定义："
                f"{edge.from_feature_id}"
            )
        if edge.to_feature_id not in feature_ids:
            errors.append(
                f"risk_graph.edges[{idx}].to_feature_id 未在 IR 中定义："
                f"{edge.to_feature_id}"
            )

    for idx, scenario in enumerate(report.test_strategy):
        if scenario.feature_id not in feature_ids:
            errors.append(
                f"test_strategy[{idx}].feature_id 未在 IR 中定义："
                f"{scenario.feature_id}"
            )

    for idx, clarification in enumerate(report.clarifications):
        if clarification.priority_rank < 1:
            errors.append(f"clarifications[{idx}].priority_rank 必须大于等于 1。")
        if clarification.related_id not in valid_related_ids:
            errors.append(
                f"clarifications[{idx}].related_id 必须引用 Gap.id 或 Feature.id："
                f"{clarification.related_id}"
            )

    for idx, reference in enumerate(report.kb_references):
        if not str(reference).strip():
            continue
        if not _reference_exists_in_kb(str(reference), kb_context):
            errors.append(f"kb_references[{idx}] 未在 kb_context 中找到：{reference}")

    return errors


def _reference_exists_in_kb(reference: str, kb_context: str) -> bool:
    """判断 KB 引用是否能在检索上下文中找到。"""
    if not kb_context.strip():
        return False
    quote = _normalise_text(reference)
    source = _normalise_text(kb_context)
    return bool(quote) and (quote in source or _quote_tokens_in_source(quote, source))


def _normalise_text(text: str) -> str:
    """规整空白和常见中英文标点，便于包含判断。"""
    normalised = str(text or "")
    replacements = {
        "“": '"',
        "”": '"',
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "、": ",",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
    }
    for src, dst in replacements.items():
        normalised = normalised.replace(src, dst)
    return re.sub(r"\s+", "", normalised)


def _quote_tokens_in_source(quote: str, source_text: str) -> bool:
    """允许 KB 引用由多个短片段组成。"""
    tokens = _text_tokens(quote)
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] in source_text
    matched = sum(1 for token in dict.fromkeys(tokens) if token in source_text)
    return matched >= 2 and matched / len(set(tokens)) >= 0.5


def _text_tokens(text: str) -> list[str]:
    """提取中文和英文数字片段用于宽松匹配。"""
    tokens: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part) and len(part) > 6:
            tokens.extend(part[idx:idx + 2] for idx in range(len(part) - 1))
        elif len(part) >= 2:
            tokens.append(part)
    return tokens


def _build_graph_from_enriched_ir(
    *,
    enriched_ir: EnrichedRequirementIR,
    report: AnalysisReport,
    module: str,
) -> dict:
    """把 EnrichedIR 与 AnalysisReport 合成为兼容旧下游的 RequirementGraph。"""
    ir = enriched_ir.ir
    review = enriched_ir.review
    return {
        "summary": ir.summary or f"{module} 需求分析",
        "actors": [actor.name for actor in ir.actors],
        "features": [
            _feature_to_graph_item(feature, report=report, enriched_ir=enriched_ir)
            for feature in ir.features
        ],
        "business_rules": [
            rule.model_dump(mode="json")
            for rule in ir.business_rules
        ],
        "state_transitions": [
            _state_machine_to_graph_item(machine)
            for machine in ir.state_machines
        ],
        "risks": [
            _review_risk_to_graph_item(risk)
            for risk in review.risks
        ],
        "clarifications": [
            _analysis_clarification_to_graph_item(item)
            for item in sorted(
                report.clarifications,
                key=lambda item: item.priority_rank,
            )
        ],
        "test_strategy": _build_compatible_test_strategy(report, enriched_ir),
        "risk_graph": report.risk_graph.model_dump(mode="json"),
        "kb_references": list(report.kb_references),
        "regression_scope": list(report.regression_scope),
    }


def _feature_to_graph_item(
    feature,
    *,
    report: AnalysisReport,
    enriched_ir: EnrichedRequirementIR,
) -> dict:
    """将 IR feature 转换为旧 RequirementGraph feature 结构。"""
    risk_level, risk_reason = _feature_risk_summary(
        feature.id,
        report=report,
        enriched_ir=enriched_ir,
    )
    related_scenarios = [
        scenario
        for scenario in report.test_strategy
        if scenario.feature_id == feature.id
    ]
    test_focus = [*feature.test_hints]
    for scenario in related_scenarios:
        if scenario.scenario:
            test_focus.append(scenario.scenario)
        test_focus.extend(scenario.focus)

    return {
        "id": feature.id,
        "name": feature.name,
        "description": feature.description,
        "priority": feature.priority,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "boundaries": list(feature.acceptance_criteria),
        "test_focus": test_focus,
        "dependencies": list(feature.dependencies),
        "notes": "",
    }


def _feature_risk_summary(
    feature_id: str,
    *,
    report: AnalysisReport,
    enriched_ir: EnrichedRequirementIR,
) -> tuple[str, str]:
    """根据 reviewer 风险和 risk_graph 给 feature 补兼容风险摘要。"""
    is_risk_node = feature_id in set(report.risk_graph.nodes)
    has_high_review_risk = any(
        risk.level == "high"
        for risk in enriched_ir.review.risks
    )
    if is_risk_node and has_high_review_risk:
        return "high", "reviewer 存在高风险标注，且该功能参与风险传导关系。"
    if is_risk_node:
        return "medium", "该功能参与 risk_graph 中的风险传导关系。"
    if enriched_ir.review.risks:
        return "medium", "存在 reviewer 风险标注，需结合 risks 列表评估。"
    return "low", ""


def _state_machine_to_graph_item(machine) -> dict:
    """将 IR 状态机转换为旧 RequirementGraph state_transitions 结构。"""
    return {
        "entity": machine.entity,
        "states": list(machine.states),
        "transitions": [
            {
                "from": transition.from_state,
                "to": transition.to_state,
                "trigger": transition.trigger,
                "condition": transition.guard,
            }
            for transition in machine.transitions
        ],
    }


def _review_risk_to_graph_item(risk) -> dict:
    """将 reviewer 风险转换为旧 RequirementGraph risks 结构。"""
    suggestion = risk.suggestion
    return {
        "area": risk.area,
        "level": risk.level,
        "description": risk.description,
        "suggestion": suggestion,
        "mitigation": suggestion,
    }


def _analysis_clarification_to_graph_item(clarification) -> dict:
    """将 AnalysisReport 澄清项转换为旧 RequirementGraph clarifications 结构。"""
    return {
        "id": clarification.related_id,
        "related_id": clarification.related_id,
        "question": clarification.question,
        "context": f"优先级：{clarification.priority_rank}",
        "impact": clarification.impact_if_unresolved,
        "priority_rank": clarification.priority_rank,
    }


def _build_compatible_test_strategy(
    report: AnalysisReport,
    enriched_ir: EnrichedRequirementIR,
) -> dict:
    """将 AnalysisReport 测试场景转换为旧 RequirementGraph test_strategy 结构。"""
    scenarios = [
        scenario.model_dump(mode="json")
        for scenario in report.test_strategy
    ]
    focus_areas = [
        f"[{scenario.feature_id}] {scenario.scenario}"
        for scenario in report.test_strategy
        if scenario.scenario.strip()
    ]
    if report.regression_scope:
        focus_areas.extend(f"回归范围：{item}" for item in report.regression_scope)

    return {
        "scope": "基于已确认 RequirementIR、reviewer 标注和 KB 回归参考生成。",
        "focus_areas": focus_areas,
        "exclusions": list(enriched_ir.ir.out_of_scope),
        "suggestion": "优先覆盖 P0 功能、reviewer 风险项与 risk_graph 中存在传导关系的路径。",
        "scenarios": scenarios,
        "regression_scope": list(report.regression_scope),
    }


def _render_analysis_validation_failure(errors: list[str]) -> str:
    """渲染 AnalysisReport 引用校验失败信息。"""
    lines = [
        "需求分析未生成结构化结果：AnalysisReport 未通过引用校验。",
        "我没有保存 JSON，因为以下字段引用了不存在的 feature/gap 或 KB 文本：",
        "",
    ]
    for idx, error in enumerate(errors[:20], start=1):
        lines.append(f"{idx}. {error}")
    return "\n".join(lines)


def _render_analysis_summary(
    data: RequirementAnalysisData,
    json_path: str = "",
) -> str:
    """渲染需求分析摘要。"""
    graph = data.graph
    features = graph.get("features", [])
    risks = graph.get("risks", [])
    clarifications = graph.get("clarifications", [])
    high_risks = [risk for risk in risks if risk.get("level") == "high"]

    lines = [f"需求分析完成：{data.module}", ""]
    if json_path:
        lines += [f"JSON 文件：{json_path}", ""]
    lines += [
        f"摘要：{data.summary}",
        (
            f"功能点：{len(features)} 个"
            f"（P0: {sum(1 for f in features if f.get('priority') == 'P0')} 个，"
            f"P1: {sum(1 for f in features if f.get('priority') == 'P1')} 个）"
        ),
        f"状态转换实体：{len(graph.get('state_transitions', []))} 个",
        f"风险点：{len(risks)} 个（高风险: {len(high_risks)} 个）",
        f"待澄清问题：{len(clarifications)} 个",
    ]

    if high_risks:
        lines += ["", "高风险区域："]
        for risk in high_risks[:3]:
            desc = risk.get("description", "")
            desc_short = desc[:60] + "..." if len(desc) > 60 else desc
            lines.append(f"  [{risk.get('area', '')}] {desc_short}")

    if clarifications:
        lines += ["", "待澄清问题（前3条）："]
        for question in clarifications[:3]:
            lines.append(f"  {question.get('id', '')}: {question.get('question', '')}")

    strategy = graph.get("test_strategy", {})
    if strategy.get("suggestion"):
        lines += ["", f"测试策略建议：{strategy['suggestion']}"]

    return "\n".join(lines)


def _build_request_metadata(request_id: str) -> dict:
    """构造产物元数据。"""
    metadata: dict[str, str] = {}
    if request_id:
        metadata["request_id"] = request_id
    return metadata
