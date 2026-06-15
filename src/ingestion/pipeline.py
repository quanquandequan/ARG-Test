"""IngestionPipeline 编排 load → clean → chunk → embed → persist 流程。"""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.core.exceptions import CollectionNotFoundError
from src.core.logging import get_logger
from src.embedding.base import BaseEmbedder
from src.ingestion.chunker import ChineseChunker, Chunk
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
    """端到端编排文档摄取流程。"""

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
        """加载、清洗并分块单个文档。"""
        doc = self._loader.load(path)
        logger.info("document_loaded", path=str(path), doc_id=doc.id)

        if doc.segments:
            # 结构化文档（Excel）：全文字符串仅用于调试，段落内容单独清洗，跳过全文清洗
            chunks = []
            for seg in doc.segments:
                cleaned = self._cleaner.clean(seg["content"])  # 对每行内容单独清洗
                # 用 Excel 行号（row_index）作为 chunk_index，与 metadata 保持一致，避免空行跳过后错位
                row_idx = seg.get("metadata", {}).get("row_index", len(chunks))
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc.id,
                    content=cleaned,
                    chunk_index=row_idx,
                    metadata=dict(seg.get("metadata", {})),
                ))
            logger.info("document_chunked_from_segments", doc_id=doc.id, chunk_count=len(chunks))
        else:
            # 非结构化文档：清洗全文后走 ChineseChunker
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
        """运行完整摄取流程，并将向量持久化到向量数据库。"""
        if self._embedder is None or self._vectordb is None:
            raise RuntimeError(
                "IngestionPipeline requires embedder and vectordb for ingest_and_store()."
            )

        doc, chunks = self.ingest(path)
        resolved_source_path = source_path or str(path.resolve())

        # 用文件绝对路径的 SHA-256 前 24 位作为稳定 document_id：
        # 同一文件每次摄取得到相同 ID，配合下面的删旧逻辑实现幂等入库
        stable_doc_id = hashlib.sha256(resolved_source_path.encode()).hexdigest()[:24]

        if not chunks:
            return PersistedIngestion(
                document=doc,
                chunks=[],
                vectors=[],
                source_path=resolved_source_path,
            )

        # 入库前删除同 document_id 的旧 chunks，防止重复入库
        try:
            deleted = self._vectordb.delete_by_document_id(stable_doc_id)
            if deleted:
                logger.info("document_old_chunks_deleted", doc_id=stable_doc_id, count=deleted)
        except CollectionNotFoundError:
            pass  # 首次入库，集合尚不存在，跳过删除

        vectors = self._embedder.embed_documents([chunk.content for chunk in chunks])
        rows = []
        source = Path(resolved_source_path)
        document_metadata = dict(doc.metadata or {})
        for chunk, vec in zip(chunks, vectors):
            rows.append(
                (
                    chunk.id,
                    stable_doc_id,  # 使用稳定 ID 替代随机 uuid，保证幂等
                    chunk.content,
                    chunk.chunk_index,
                    vec,
                    {
                        **document_metadata,
                        "source_path": resolved_source_path,
                        "source_name": source.name,
                        "source_ext": source.suffix.lower(),
                        "source_format": str(
                            document_metadata.get("format") or source.suffix.lstrip(".")
                        ).lower(),
                        "chunk_index": chunk.chunk_index,
                        **dict(chunk.metadata or {}),
                    },
                )
            )

        self._vectordb.insert(rows)
        logger.info("document_persisted", doc_id=stable_doc_id, chunk_count=len(chunks))
        return PersistedIngestion(
            document=doc,
            chunks=chunks,
            vectors=vectors,
            source_path=resolved_source_path,
        )
