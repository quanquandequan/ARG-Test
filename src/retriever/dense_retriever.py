"""稠密向量检索器：embed query -> search Milvus。"""

from src.core.logging import get_logger
from src.embedding.base import BaseEmbedder
from src.vectordb.base import BaseVectorDB, SearchResult

logger = get_logger(__name__)


class DenseRetriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        top_k: int = 20,
        similarity_threshold: float = 0.0,
    ):
        self._embedder = embedder
        self._vectordb = vectordb
        self._top_k = top_k
        self._threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_vec = self._embedder.embed_query(query)
        results = self._vectordb.search(
            query_vec, top_k=top_k or self._top_k, filters=filters
        )
        if self._threshold > 0:
            kept = [r for r in results if r.score >= self._threshold]
            dropped = len(results) - len(kept)
            if dropped:
                logger.debug(
                    "retrieval_threshold_filter",
                    threshold=self._threshold,
                    kept=len(kept),
                    dropped=dropped,
                )
            if kept or not results:
                results = kept
            else:
                logger.warning(
                    "retrieval_threshold_all_dropped_keep_original",
                    threshold=self._threshold,
                    candidates=len(results),
                    max_score=max((r.score for r in results), default=0.0),
                )
        return results
