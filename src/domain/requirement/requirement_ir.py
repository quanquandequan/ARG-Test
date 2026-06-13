"""RequirementIR：需求文档的类型化中间表示。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import field_validator, model_validator

from pydantic import BaseModel, Field


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
        """解析 LLM JSON 输出（去除 markdown 代码块）；失败时返回 None。"""
        import json
        import re

        text = re.sub(r"```(?:json)?\s*", "", raw, count=1, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, count=1, flags=re.MULTILINE).strip()

        candidates = [text]
        extracted = _find_first_json_object(text)
        if extracted:
            candidates.append(extracted)
        
        for candidate in candidates:
            if not candidate:
                continue
            # Try direct parse first
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return cls.model_validate(data)
            except Exception:
                pass
        return None


def _find_first_json_object(text: str) -> str:
    """从文本中提取第一个顶层 {...} 块。"""
    import re

    match = re.search(r"\{[\s\S]*\}", text)
    return match.group() if match else ""


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
        import json
        import re

        text = re.sub(r"```(?:json)?\s*", "", raw, count=1, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, count=1, flags=re.MULTILINE).strip()

        # Try to find a valid JSON object
        # _find_first_json_object goes first (handles ```json fences better)
        candidates = []
        extracted = _find_first_json_object(text)
        if extracted:
            candidates.append(extracted)
        candidates.append(text)
        
        for candidate in candidates:
            if not candidate:
                continue
            # Try 1: direct parse
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return cls.model_validate(data)
            except Exception:
                pass
            # Try 2: replace ASCII double quotes used as Chinese quotes inside strings
            # Pattern: Chinese char + " + non-quote content + " + Chinese/punctuation char
            repaired = re.sub(
                r'([\u4e00-\u9fff])"([^"]{1,50})"([\u4e00-\u9fff\uff0c\u3002\uff1b\uff0e])',
                r'\1「\2」\3',
                candidate
            )
            if repaired != candidate:
                try:
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        return cls.model_validate(data)
                except Exception:
                    pass
        return None

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
    def from_llm_json(cls, raw: str) -> "AnalysisReport | None":
        """解析 LLM 输出的 AnalysisReport JSON；失败时返回 None。"""
        import json
        import re

        text = re.sub(r"```(?:json)?\s*", "", raw, count=1, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, count=1, flags=re.MULTILINE).strip()

        # 多策略提取 JSON
        candidates = []
        # 策略1: 从 markdown fence 中提取
        extracted = _find_first_json_object(text)
        if extracted:
            candidates.append(extracted)
        # 策略2: 直接解析（fence 被正则剥离后的纯 JSON）
        candidates.append(text)
        # 策略3: 从原始输入中提取（跳过 fence 剥离，直接从 raw 找）
        raw_extracted = _find_first_json_object(raw)
        if raw_extracted and raw_extracted not in candidates:
            candidates.append(raw_extracted)

        for candidate in candidates:
            if not candidate:
                continue
            # 直接解析
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    data = _coerce_string_lists(data, "kb_references", "regression_scope")
                    report = cls.model_validate(data)
                    report.graph = data
                    return report
            except Exception:
                pass
            # 修复常见 LLM JSON 错误后重试
            repaired = candidate.replace("\\'", "'")
            repaired = re.sub(
                r'([一-鿿])"([^"]{1,50})"([一-鿿，。；．])',
                r'\1「\2」\3',
                repaired,
            )
            if repaired != candidate:
                try:
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        report = cls.model_validate(data)
                        report.graph = data
                        return report
                except Exception:
                    pass
        return None

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
