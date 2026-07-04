"""使用 pypdf 抽取文本的 PDF 文档 reader。"""

import uuid
from pathlib import Path

from pypdf import PdfReader as PyPDFReader

from src.core.exceptions import IngestionError
from src.ingestion.readers.base import BaseReader, Document


class PDFReader(BaseReader):
    def read(self, path: Path) -> Document:
        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        try:
            reader = PyPDFReader(str(path))
        except Exception as e:
            raise IngestionError(f"Failed to open PDF: {path} — {e}") from e

        pages: list[str] = []
        metadata: dict = {
            "page_count": len(reader.pages),
            "sections_by_page": {},
        }

        if reader.metadata:
            meta = reader.metadata
            if meta.title:
                metadata["title"] = meta.title
            if meta.author:
                metadata["author"] = meta.author

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(text)
            metadata["sections_by_page"][i + 1] = text[:120]

        content = "\n\n".join(pages)

        return Document(
            id=str(uuid.uuid4()),
            source_path=str(path.resolve()),
            content=content,
            metadata=metadata,
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
