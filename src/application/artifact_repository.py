"""Artifact persistence abstraction with a local filesystem implementation."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.domain.artifacts import ArtifactKind, ArtifactRecord

_OUTPUT_DIRS = {
    ArtifactKind.REQUIREMENT_ANALYSIS_JSON: "requirements",
    ArtifactKind.REQUIREMENT_ANALYSIS_MARKDOWN: "requirements",
    ArtifactKind.REQUIREMENT_IR_JSON: "requirement_ir",
    ArtifactKind.REQUIREMENT_IR_MARKDOWN: "requirement_ir",
    ArtifactKind.REQUIREMENT_REVIEW_JSON: "requirement_ir",
    ArtifactKind.REQUIREMENT_REVIEW_MARKDOWN: "requirement_ir",
    ArtifactKind.TEST_CASES_XLSX: "test_cases",
}


class LocalArtifactRepository:
    """Store generated artifacts on the local filesystem."""

    def __init__(self, base_dir: str = "./outputs"):
        self._base_dir = Path(base_dir)

    def allocate(
        self,
        kind: ArtifactKind,
        module: str,
        extension: str,
        metadata: dict[str, Any] | None = None,
        suffix: str = "",
    ) -> ArtifactRecord:
        safe_module = re.sub(r'[\\/:*?"<>|]', "_", module.strip() or "artifact")
        artifact_id = str(uuid.uuid4())
        subdir = _OUTPUT_DIRS.get(kind, "misc")
        directory = self._base_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)

        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"{safe_module}{suffix_part}_{artifact_id[:8]}{extension}"
        path = directory / filename

        return ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            path=path.resolve(),
            media_type=_infer_media_type(extension),
            metadata=dict(metadata or {}),
        )

    def save_text(
        self,
        kind: ArtifactKind,
        module: str,
        text: str,
        extension: str,
        metadata: dict[str, Any] | None = None,
        suffix: str = "",
    ) -> ArtifactRecord:
        artifact = self.allocate(kind, module, extension, metadata=metadata, suffix=suffix)
        artifact.path.write_text(text, encoding="utf-8")
        return artifact

    def save_json(
        self,
        kind: ArtifactKind,
        module: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
        suffix: str = "",
    ) -> ArtifactRecord:
        artifact = self.allocate(kind, module, ".json", metadata=metadata, suffix=suffix)
        artifact.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact

    def finalize(
        self,
        artifact: ArtifactRecord,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        if metadata is None:
            return artifact
        merged = dict(artifact.metadata)
        merged.update(metadata)
        return replace(artifact, metadata=merged)


def _infer_media_type(extension: str) -> str:
    mapping = {
        ".json": "application/json",
        ".md": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return mapping.get(extension.lower(), "application/octet-stream")

