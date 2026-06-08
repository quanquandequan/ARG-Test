"""JSON exporter for TestDesignArtifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.application.exporters.common import (
    case_to_dict,
    maybe_json,
    normalise_expected_visibility,
)
from src.domain.artifacts.test_design_artifact import TestDesignArtifact


class JsonExporter:
    """Export automation-friendly structured JSON from test design artifacts."""

    def export(self, artifact: TestDesignArtifact, path: Path) -> None:
        payload = self.build_payload(artifact)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def build_payload(self, artifact: TestDesignArtifact) -> dict:
        prefix = re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", artifact.module)[:4].upper()
        prefix = prefix or "TC"
        cases = []
        for idx, case in enumerate(artifact.test_cases, start=1):
            row_case = case_to_dict(case)
            case_id = row_case.get("id") or f"{prefix}-{str(idx).zfill(3)}"
            cases.append({
                "id": case_id,
                "mode": "automation",
                "title": row_case.get("title", ""),
                "module": row_case.get("module", artifact.module),
                "business_name": row_case.get("business_name", ""),
                "ui_display_name": row_case.get("ui_display_name", ""),
                "data_precondition": row_case.get("data_setup", ""),
                "precondition": row_case.get("precondition", ""),
                "page_route": maybe_json(row_case.get("page_route", "")),
                "expected_visibility": normalise_expected_visibility(
                    row_case.get("expected_visibility", "")
                ),
                "locator_chain": maybe_json(row_case.get("locator_chain", "")),
                "anchor_text": row_case.get("anchor_text", ""),
                "search_strategy": maybe_json(row_case.get("search_strategy", "")),
                "selectors": maybe_json(row_case.get("selectors", "")),
                "automation_steps": maybe_json(row_case.get("automation_steps", "")),
                "assertions": maybe_json(row_case.get("assertions", "")),
                "forbidden_locators": maybe_json(row_case.get("forbidden_locators", "")),
                "steps": row_case.get("steps", ""),
                "expected": row_case.get("expected", ""),
                "priority": row_case.get("priority", "P1"),
                "type": row_case.get("type", "正向"),
                "notes": row_case.get("notes", ""),
            })
        return {
            "schema_version": "1.0",
            "module": artifact.module,
            "generation_mode": artifact.generation_mode,
            "case_count": len(cases),
            "cases": cases,
            "scenario_count": len(artifact.scenarios),
            "test_point_count": len(artifact.test_points),
        }
