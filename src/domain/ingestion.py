"""Typed DTOs for document ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    chunk_count: int
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

