"""IngestionPipeline orchestrates load → clean → chunk → embed → persist."""

from dataclasses import dataclass
from pathlib import Path

from src.core.logging import get_logger
from src.embedding.base import BaseEmbedder
from src.ingestion.chunker import Chunk, ChineseChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.readers.base import Document
from src.vectordb.base import BaseVectorDB

logger = get_logger(__name__)


@dataclass(slots=True)
class PersistedIngestion:
    document: Document
    chunks: list[Chunk]
    vectors: list | object
    source_path: str


class IngestionPipeline:
    """Orchestrate document ingestion end-to-end."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        cleaner: TextCleaner | None = None,
        chunker: ChineseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        vectordb: BaseVectorDB | None = None,
    ):
        self._loader = loader or DocumentLoader()
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or ChineseChunker()
        self._embedder = embedder
        self._vectordb = vectordb

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

    def ingest_and_store(
        self,
        path: Path,
        source_path: str | None = None,
    ) -> PersistedIngestion:
        """Run the full ingestion pipeline and persist vectors to the vector DB."""
        if self._embedder is None or self._vectordb is None:
            raise RuntimeError(
                "IngestionPipeline requires embedder and vectordb for ingest_and_store()."
            )

        doc, chunks = self.ingest(path)
        if not chunks:
            return PersistedIngestion(
                document=doc,
                chunks=[],
                vectors=[],
                source_path=source_path or str(path),
            )

        vectors = self._embedder.embed_documents([chunk.content for chunk in chunks])
        rows = []
        for chunk, vec in zip(chunks, vectors):
            rows.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    vec,
                    {
                        "source_path": source_path or str(path),
                        "chunk_index": chunk.chunk_index,
                        **dict(chunk.metadata or {}),
                    },
                )
            )

        self._vectordb.insert(rows)
        logger.info("document_persisted", doc_id=doc.id, chunk_count=len(chunks))
        return PersistedIngestion(
            document=doc,
            chunks=chunks,
            vectors=vectors,
            source_path=source_path or str(path),
        )
