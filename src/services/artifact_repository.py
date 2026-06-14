"""产物持久化抽象及本地文件系统实现。"""

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
    ArtifactKind.REQUIREMENT_IR_JSON: "requirement_ir",
    ArtifactKind.REQUIREMENT_IR_MARKDOWN: "requirement_ir",
    ArtifactKind.REQUIREMENT_REVIEW_JSON: "requirement_ir",
    ArtifactKind.REQUIREMENT_REVIEW_MARKDOWN: "requirement_ir",
    ArtifactKind.TEST_CASES_XLSX: "test_cases",
    ArtifactKind.TEST_CASES_AUTOMATION_JSON: "test_cases",
    ArtifactKind.EXECUTION_REPORT_JSON: "test_execution",
    ArtifactKind.EXECUTION_SCREENSHOT_PNG: "test_execution",
}


class LocalArtifactRepository:
    """将生成的产物存储到本地文件系统。"""

    def __init__(self, base_dir: str = "./outputs"):
        self._base_dir = Path(base_dir)

    def allocate(
        self,
        kind: ArtifactKind,
        module: str,
        extension: str,
        metadata: dict[str, Any] | None = None,
        suffix: str = "",
        directory: str | Path | None = None,
    ) -> ArtifactRecord:
        safe_module = re.sub(r'[\\/:*?"<>|]', "_", module.strip() or "artifact")
        artifact_id = str(uuid.uuid4())
        if directory is None:
            subdir = _OUTPUT_DIRS.get(kind, "misc")
            target_dir = self._base_dir / subdir
        else:
            target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"{safe_module}_{artifact_id[:8]}{suffix_part}{extension}"
        path = target_dir / filename

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
        directory: str | Path | None = None,
    ) -> ArtifactRecord:
        artifact = self.allocate(
            kind,
            module,
            extension,
            metadata=metadata,
            suffix=suffix,
            directory=directory,
        )
        artifact.path.write_text(text, encoding="utf-8")
        return artifact

    def save_json(
        self,
        kind: ArtifactKind,
        module: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
        suffix: str = "",
        directory: str | Path | None = None,
    ) -> ArtifactRecord:
        artifact = self.allocate(
            kind,
            module,
            ".json",
            metadata=metadata,
            suffix=suffix,
            directory=directory,
        )
        artifact.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact

    def save_bytes(
        self,
        kind: ArtifactKind,
        module: str,
        payload: bytes,
        extension: str,
        metadata: dict[str, Any] | None = None,
        suffix: str = "",
        directory: str | Path | None = None,
    ) -> ArtifactRecord:
        artifact = self.allocate(
            kind,
            module,
            extension,
            metadata=metadata,
            suffix=suffix,
            directory=directory,
        )
        artifact.path.write_bytes(payload)
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
        ".png": "image/png",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return mapping.get(extension.lower(), "application/octet-stream")
