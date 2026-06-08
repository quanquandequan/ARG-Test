"""应用服务与工具共享的类型化产物元数据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ArtifactKind(StrEnum):
    REQUIREMENT_ANALYSIS_JSON = "requirement_analysis_json"
    REQUIREMENT_IR_JSON = "requirement_ir_json"
    REQUIREMENT_IR_MARKDOWN = "requirement_ir_markdown"
    REQUIREMENT_REVIEW_JSON = "requirement_review_json"
    REQUIREMENT_REVIEW_MARKDOWN = "requirement_review_markdown"
    TEST_CASES_XLSX = "test_cases_xlsx"
    TEST_CASES_AUTOMATION_JSON = "test_cases_automation_json"
    EXECUTION_REPORT_JSON = "execution_report_json"
    EXECUTION_SCREENSHOT_PNG = "execution_screenshot_png"


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
