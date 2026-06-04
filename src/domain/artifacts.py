"""Typed artifact metadata shared by application services and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactKind(str, Enum):
    REQUIREMENT_ANALYSIS_JSON = "requirement_analysis_json"
    REQUIREMENT_ANALYSIS_MARKDOWN = "requirement_analysis_markdown"
    REQUIREMENT_IR_JSON = "requirement_ir_json"
    REQUIREMENT_IR_MARKDOWN = "requirement_ir_markdown"
    REQUIREMENT_REVIEW_JSON = "requirement_review_json"
    REQUIREMENT_REVIEW_MARKDOWN = "requirement_review_markdown"
    TEST_CASES_XLSX = "test_cases_xlsx"


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    kind: ArtifactKind
    path: Path
    media_type: str
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

