"""IngestionPipeline orchestrates load → clean → chunk."""

from pathlib import Path

from src.core.logging import get_logger
from src.ingestion.chunker import Chunk, ChineseChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.readers.base import Document

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrate document ingestion end-to-end."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        cleaner: TextCleaner | None = None,
        chunker: ChineseChunker | None = None,
    ):
        self._loader = loader or DocumentLoader()
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or ChineseChunker()

    def ingest(self, path: Path) -> tuple[Document, list[Chunk]]:
        """Load, clean, and chunk a single document."""
        doc = self._loader.load(path)
        logger.info("document_loaded", path=str(path), doc_id=doc.id)

        doc.content = self._cleaner.clean(doc.content)
        logger.debug("document_cleaned", doc_id=doc.id, length=len(doc.content))

        chunks = self._chunker.split(doc.id, doc.content)
        logger.info("document_chunked", doc_id=doc.id, chunk_count=len(chunks))

        return doc, chunks

    def ingest_batch(self, paths: list[Path]) -> list[tuple[Document, list[Chunk]]]:
        results: list[tuple[Document, list[Chunk]]] = []
        for p in paths:
            results.append(self.ingest(p))
        return results
