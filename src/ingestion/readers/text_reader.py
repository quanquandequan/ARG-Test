"""Plain text reader with encoding detection."""

import uuid
from pathlib import Path

import chardet

from src.core.exceptions import IngestionError
from src.ingestion.readers.base import BaseReader, Document


class TextReader(BaseReader):
    def read(self, path: Path) -> Document:
        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8") or "utf-8"

        try:
            content = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            content = raw_bytes.decode("utf-8", errors="replace")

        # Strip BOM if present
        if content and content[0] == "﻿":
            content = content[1:]

        return Document(
            id=str(uuid.uuid4()),
            source_path=str(path.resolve()),
            content=content,
            metadata={"encoding": encoding},
        )

    def supported_extensions(self) -> list[str]:
        return [".txt", ".log", ".csv", ".json", ".xml", ".html"]
