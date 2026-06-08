"""RequirementIR：需求文档的类型化中间表示。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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

        text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        for candidate in [text, _find_first_json_object(text)]:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return cls.model_validate(data)
            except (json.JSONDecodeError, Exception):
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
    suggestion: str


class Gap(BaseModel):
    """设计或执行测试所需但缺失的信息。"""

    id: str
    description: str
    impact: str
    question: str


class ReviewRisk(BaseModel):
    area: str
    level: Literal["high", "medium", "low"] = "medium"
    description: str
    mitigation: str


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

        text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        for candidate in [text, _find_first_json_object(text)]:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return cls.model_validate(data)
            except (json.JSONDecodeError, Exception):
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
