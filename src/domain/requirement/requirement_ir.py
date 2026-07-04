"""RequirementIR：需求文档的类型化中间表示。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Actor(BaseModel):
    """与功能交互的人、系统或角色。"""

    name: str
    role: str


class Feature(BaseModel):
    """面向用户的独立能力或系统行为。"""

    id: str
    name: str
    description: str
    priority: Literal["P0", "P1", "P2"] = "P1"
    acceptance_criteria: list[str] = Field(default_factory=list)
    test_hints: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class BusinessRule(BaseModel):
    """系统必须始终遵守的约束或不变量。"""

    id: str
    description: str
    condition: str
    outcome: str
    related_features: list[str] = Field(default_factory=list)


class StateTransition(BaseModel):
    from_state: str
    to_state: str
    trigger: str
    guard: str = ""


class StateMachine(BaseModel):
    """具备清晰生命周期的领域实体状态图。"""

    entity: str
    states: list[str]
    initial_state: str = ""
    transitions: list[StateTransition] = Field(default_factory=list)


class DataField(BaseModel):
    name: str
    field_type: str = "string"
    constraints: list[str] = Field(default_factory=list)
    required: bool = True


class DataEntity(BaseModel):
    """与功能相关的数据对象（请求体、DB 记录、表单）。"""

    name: str
    fields: list[DataField] = Field(default_factory=list)


class RequirementIR(BaseModel):
    """需求文档的结构化中间表示。"""

    version: str = "1.0"
    module: str
    summary: str

    actors: list[Actor] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    business_rules: list[BusinessRule] = Field(default_factory=list)
    state_machines: list[StateMachine] = Field(default_factory=list)
    data_entities: list[DataEntity] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    source_length: int = 0
    has_kb_context: bool = False

    def feature_count(self) -> int:
        return len(self.features)

    def p0_features(self) -> list[Feature]:
        return [f for f in self.features if f.priority == "P0"]

    def to_compact_summary(self) -> str:
        """单行统计信息，便于嵌入 Agent 工具输出。"""
        return (
            f"功能点 {len(self.features)} 个"
            f"（P0:{len(self.p0_features())}）"
            f"｜业务规则 {len(self.business_rules)} 条"
            f"｜状态机 {len(self.state_machines)} 个"
            f"｜数据实体 {len(self.data_entities)} 个"
        )

    @classmethod
    def from_llm_json(cls, raw: str) -> RequirementIR | None:
        """解析 LLM JSON 输出（去除 markdown 代码块 + 中文引号修复）；失败时返回 None。"""
        return _parse_llm_model_json(raw, cls.model_validate)


def _find_first_json_object(text: str) -> str:
    """从文本中提取第一个顶层 {...} 块。"""
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group() if match else ""


def _repair_llm_json_quotes(text: str) -> str:
    """修复 LLM 常见的 JSON 转义错误。

    需求文档里大量引用 UI 文案（如"加追""追番表"），LLM 输出 JSON 时容易把
    这类中文引号误写成和外层字符串相同的 ASCII 双引号，导致字符串提前截断、
    破坏 JSON 结构。这里把"中文字符 + 引号包裹的短片段 + 中文字符/标点"
    这种模式统一替换成全角「」，避免破坏外层 JSON 转义。
    """
    repaired = text.replace("\\'", "'")
    return re.sub(
        r'([\u4e00-\u9fff])"([^"]{1,50})"([\u4e00-\u9fff，。；．])',
        r'\1「\2」\3',
        repaired,
    )


def _parse_llm_model_json(raw: str, validate):
    """多策略解析 LLM 输出为 JSON 对象后调用 ``validate(data)`` 得到模型实例。

    RequirementIR / ReviewResult / AnalysisReport 三个 ``from_llm_json`` 共用此逻辑，
    避免各自实现导致修复策略不一致（历史上 RequirementIR 就因为缺少中文引号修复，
    只要需求文档引用了 UI 文案就必现解析失败）。

    候选顺序：从去 fence 文本中提取的 JSON 对象 -> 整个去 fence 文本 -> 从原始
    未去 fence 文本中提取的 JSON 对象（兜底极端 fence 场景）；每个候选都先直接
    解析，失败后再用中文引号修复规则重试一次。``validate`` 抛出的异常也视为
    "当前候选不合法"，会继续尝试下一个候选，而不是直接判定解析失败。
    """
    text = re.sub(r"```(?:json)?\s*", "", raw, count=1, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, count=1, flags=re.MULTILINE).strip()

    candidates = []
    extracted = _find_first_json_object(text)
    if extracted:
        candidates.append(extracted)
    candidates.append(text)
    raw_extracted = _find_first_json_object(raw)
    if raw_extracted and raw_extracted not in candidates:
        candidates.append(raw_extracted)

    for candidate in candidates:
        if not candidate:
            continue
        variants = [candidate]
        repaired = _repair_llm_json_quotes(candidate)
        if repaired != candidate:
            variants.append(repaired)
        for variant in variants:
            try:
                data = json.loads(variant)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            try:
                return validate(data)
            except Exception:
                continue
    return None


class Ambiguity(BaseModel):
    """需求中含糊、不清晰或主观的表述。"""

    id: str
    location: str
    description: str
    suggestion: str = ""


class Gap(BaseModel):
    """设计或执行测试所需但缺失的信息。"""

    id: str
    description: str
    impact: str = ""
    question: str = ""


class ReviewRisk(BaseModel):
    area: str
    level: Literal["high", "medium", "low"] = "medium"
    description: str
    suggestion: str = Field(default="", alias="mitigation")
    
    @model_validator(mode="before")
    @classmethod
    def normalise_fields(cls, data: dict) -> dict:
        if isinstance(data, dict):
            # Accept both "suggestion" and "mitigation" from LLM output
            if "suggestion" in data and "mitigation" not in data:
                data["mitigation"] = data.pop("suggestion")
        return data


class ReviewResult(BaseModel):
    """``requirement_reviewer`` 工具的输出。"""

    overall_quality: Literal[
        "good", "needs_clarification", "poor"
    ] = "needs_clarification"
    score: int = Field(default=70, ge=0, le=100)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    risks: list[ReviewRisk] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @classmethod
    def from_llm_json(cls, raw: str) -> ReviewResult | None:
        """解析 LLM JSON 输出（去除 markdown 代码块 + 中文引号修复）；失败时返回 None。"""
        return _parse_llm_model_json(raw, cls.model_validate)

    def to_compact_summary(self) -> str:
        quality_emoji = {"good": "✅", "needs_clarification": "⚠️", "poor": "❌"}
        emoji = quality_emoji.get(self.overall_quality, "")
        return (
            f"{emoji} 需求质量评分：{self.score}/100（{self.overall_quality}）"
            f"｜歧义 {len(self.ambiguities)} 条"
            f"｜缺口 {len(self.gaps)} 条"
            f"｜风险 {len(self.risks)} 项"
        )


class EnrichedRequirementIR(BaseModel):
    """RequirementIR 与 ReviewResult 的合并体，供 graph_analyzer 消费。"""

    ir: RequirementIR
    review: ReviewResult

    def to_graph_analyzer_json(self) -> str:
        """序列化为传给 graph_analyzer 的 JSON 字符串。"""
        return self.model_dump_json(indent=2)

    def high_priority_gaps(self) -> list[Gap]:
        """返回影响说明非空的缺口，便于分析器优先排序。"""
        return [gap for gap in self.review.gaps if gap.impact.strip()]

    def high_risks(self) -> list[ReviewRisk]:
        """返回 reviewer 标注的高风险项。"""
        return [risk for risk in self.review.risks if risk.level == "high"]


class RiskEdge(BaseModel):
    """功能间的风险传导关系，只描述关系，不重复列举风险事实。"""

    from_feature_id: str
    to_feature_id: str
    risk_type: str = "other"
    description: str


class RiskGraph(BaseModel):
    """功能间风险传导图。"""

    nodes: list[str] = Field(default_factory=list)
    edges: list[RiskEdge] = Field(default_factory=list)


class TestScenario(BaseModel):
    """graph_analyzer 推导的测试场景，必须引用 IR 中已有 feature。"""

    feature_id: str = ""
    scenario: str = ""
    priority: Literal["P0", "P1", "P2"] = "P1"
    test_type: str = ""
    focus: list[str] = Field(default_factory=list)


class AnalysisClarification(BaseModel):
    """基于 reviewer 缺口和风险关系排序后的澄清项。"""

    priority_rank: int = 1
    related_id: str = ""
    question: str = ""
    impact_if_unresolved: str = ""



def _coerce_string_lists(data: dict, *field_names: str) -> dict:
    """将 LLM 可能输出为 list[dict] 的字段强制转为 list[str]。
    对每个 dict 元素拼接为 "{key1}: {value1} - {key2}: {value2}" 格式。
    已是字符串的元素保持不变。
    """
    import json as _json
    for name in field_names:
        values = data.get(name)
        if not isinstance(values, list):
            continue
        coerced: list[str] = []
        for item in values:
            if isinstance(item, str):
                coerced.append(item)
            elif isinstance(item, dict):
                parts = [
                    f"{k}: {v}" for k, v in item.items()
                    if v and k not in ("_", "")
                ]
                coerced.append(" - ".join(parts) if parts else _json.dumps(item, ensure_ascii=False))
            else:
                coerced.append(str(item))
        data[name] = coerced
    return data


class AnalysisReport(BaseModel):
    """graph_analyzer 的增量分析结果。"""

    risk_graph: RiskGraph = Field(default_factory=RiskGraph)
    test_strategy: list[TestScenario] = Field(default_factory=list)
    clarifications: list[AnalysisClarification] = Field(default_factory=list)
    kb_references: list[Any] = Field(default_factory=list)
    regression_scope: list[Any] = Field(default_factory=list)
    graph: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @classmethod
    def from_llm_json(cls, raw: str) -> AnalysisReport | None:
        """解析 LLM 输出的 AnalysisReport JSON（去除 markdown 代码块 + 中文引号修复）；
        失败时返回 None。"""

        def _validate(data: dict) -> AnalysisReport:
            coerced = _coerce_string_lists(data, "kb_references", "regression_scope")
            report = cls.model_validate(coerced)
            report.graph = coerced
            return report

        return _parse_llm_model_json(raw, _validate)

    @property
    def feature_count(self) -> int:
        """返回风险图中涉及的功能节点数。"""
        return len(self.risk_graph.nodes)

    @property
    def risk_count(self) -> int:
        """返回风险传导边数量。"""
        return len(self.risk_graph.edges)

    @property
    def clarification_count(self) -> int:
        """返回分析器排序后的澄清项数量。"""
        return len(self.clarifications)
