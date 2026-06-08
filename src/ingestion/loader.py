"""支持格式分发和多进程的文档加载器。"""

from pathlib import Path

from src.core.exceptions import IngestionError
from src.ingestion.readers.base import BaseReader, Document
from src.ingestion.readers.markdown_reader import MarkdownReader
from src.ingestion.readers.pdf_reader import PDFReader
from src.ingestion.readers.text_reader import TextReader
from src.ingestion.readers.xlsx_reader import XlsxReader
from src.ingestion.readers.xmind_reader import XmindReader


class DocumentLoader:
    """根据扩展名分发到合适的 reader 来加载文档。"""

    def __init__(self, readers: list[BaseReader] | None = None):
        self._readers = readers or [
            PDFReader(),
            MarkdownReader(),
            TextReader(),
            XlsxReader(),
            XmindReader(),
        ]
        self._ext_map: dict[str, BaseReader] = {}
        for r in self._readers:
            for ext in r.supported_extensions():
                self._ext_map[ext.lower()] = r

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        if not path.is_file():
            raise IngestionError(f"Not a file: {path}")

        ext = path.suffix.lower()
        reader = self._ext_map.get(ext)
        if reader is None:
            raise IngestionError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self._ext_map.keys())}"
            )
        return reader.read(path)

    def load_many(self, paths: list[Path]) -> list[Document]:
        documents: list[Document] = []
        errors: list[tuple[Path, str]] = []

        for p in paths:
            try:
                documents.append(self.load(p))
            except IngestionError as e:
                errors.append((p, str(e)))

        if errors:
            error_lines = "\n".join(f"  {p}: {e}" for p, e in errors)
            raise IngestionError(
                f"Failed to load {len(errors)}/{len(paths)} files:\n{error_lines}"
            )

        return documents
