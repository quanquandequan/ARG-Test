"""端到端文档摄取应用服务。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.domain.ingestion import IngestionResult
from src.embedding.base import BaseEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.vectordb.base import BaseVectorDB


class DocumentIngestionService:
    def __init__(
        self,
        pipeline: IngestionPipeline,
        vectordb: BaseVectorDB,
        embedder: BaseEmbedder,
    ):
        self._pipeline = pipeline
        self._vectordb = vectordb
        self._embedder = embedder

    async def ingest_upload(self, filename: str, content: bytes) -> IngestionResult:
        suffix = Path(filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = self._pipeline.ingest_and_store(
                tmp_path,
                source_path=filename or str(tmp_path),
            )
            return IngestionResult(
                document_id=result.document.id,
                chunk_count=len(result.chunks),
                source_path=result.source_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def delete_document(self, document_id: str) -> int:
        return self._vectordb.delete_by_document_id(document_id)
