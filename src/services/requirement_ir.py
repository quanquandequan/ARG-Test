"""RequirementIR — typed intermediate representation of a requirements document.

RequirementIR is the canonical output of the ``requirement_parser`` tool and
the primary input for downstream tools in the Test Design domain:

    requirement_parser → RequirementIR
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
             reviewer  point_gen   scenario_gen

Design goals:
  - Strongly typed: every field has a clear meaning for test-design tools
  - JSON-round-trippable: serialise with ``RequirementIR.model_dump_json()``
  - Pydantic v2: automatic validation when parsing LLM output
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ── Sub-models ────────────────────────────────────────────────────────────────

class Actor(BaseModel):
    """A person, system, or role that interacts with the feature."""

    name: str
    role: str


class Feature(BaseModel):
    """A discrete user-facing capability or system behaviour."""

    id: str                          # e.g. "F001"
    name: str
    description: str
    priority: Literal["P0", "P1", "P2"] = "P1"
    acceptance_criteria: list[str] = Field(default_factory=list)
    test_hints: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # other Feature IDs


class BusinessRule(BaseModel):
    """A constraint or invariant the system must always enforce."""

    id: str                          # e.g. "R001"
    description: str
    condition: str                   # "IF ..." — when the rule applies
    outcome: str                     # "THEN ..." — what must happen
    related_features: list[str] = Field(default_factory=list)


class StateTransition(BaseModel):
    from_state: str
    to_state: str
    trigger: str
    guard: str = ""                  # guard condition; empty = always valid


class StateMachine(BaseModel):
    """State diagram for a domain entity with clear lifecycle."""

    entity: str
    states: list[str]
    initial_state: str = ""
    transitions: list[StateTransition] = Field(default_factory=list)


class DataField(BaseModel):
    name: str
    field_type: str = "string"       # string / integer / boolean / enum / ...
    constraints: list[str] = Field(default_factory=list)  # e.g. "max_length=20"
    required: bool = True


class DataEntity(BaseModel):
    """A data object (request body, DB record, form) relevant to the features."""

    name: str
    fields: list[DataField] = Field(default_factory=list)


# ── Top-level IR ──────────────────────────────────────────────────────────────

class RequirementIR(BaseModel):
    """Structured intermediate representation of a requirements document.

    Produced by ``requirement_parser``, consumed by ``requirement_reviewer``,
    ``test_point_generator``, ``test_scenario_generator``, etc.
    """

    version: str = "1.0"
    module: str
    summary: str

    actors: list[Actor] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    business_rules: list[BusinessRule] = Field(default_factory=list)
    state_machines: list[StateMachine] = Field(default_factory=list)
    data_entities: list[DataEntity] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

    # Generation metadata
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    source_length: int = 0
    has_kb_context: bool = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def feature_count(self) -> int:
        return len(self.features)

    def p0_features(self) -> list[Feature]:
        return [f for f in self.features if f.priority == "P0"]

    def to_compact_summary(self) -> str:
        """One-line stats — useful for embedding in Agent tool output."""
        return (
            f"功能点 {len(self.features)} 个"
            f"（P0:{len(self.p0_features())}）"
            f"｜业务规则 {len(self.business_rules)} 条"
            f"｜状态机 {len(self.state_machines)} 个"
            f"｜数据实体 {len(self.data_entities)} 个"
        )

    @classmethod
    def from_llm_json(cls, raw: str) -> RequirementIR | None:
        """Parse LLM JSON output (strips markdown fences); returns None on failure."""
        import json
        import re

        text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        # Try direct parse
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
    """Extract the first top-level {...} block from text."""
    import re
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group() if match else ""


# ── Review models ─────────────────────────────────────────────────────────────

class Ambiguity(BaseModel):
    """A vague, unclear, or subjective statement in the requirements."""

    id: str                          # e.g. "A001"
    location: str                    # feature / rule ID this belongs to
    description: str
    suggestion: str                  # how to clarify


class Gap(BaseModel):
    """A missing piece of information needed to design or run tests."""

    id: str                          # e.g. "G001"
    description: str
    impact: str                      # what testing problem it causes
    question: str                    # concrete question to ask stakeholders


class ReviewRisk(BaseModel):
    area: str
    level: Literal["high", "medium", "low"] = "medium"
    description: str
    mitigation: str                  # suggested test approach to manage risk


class ReviewResult(BaseModel):
    """Output of the ``requirement_reviewer`` tool."""

    overall_quality: Literal["good", "needs_clarification", "poor"] = "needs_clarification"
    score: int = Field(default=70, ge=0, le=100)  # 0-100 readiness for test design
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
