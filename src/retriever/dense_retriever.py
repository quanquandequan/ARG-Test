"""Dense vector retriever — embed query → search Milvus."""

from src.core.config import get_config
from src.core.logging import get_logger
from src.embedding.base import BaseEmbedder
from src.vectordb.base import BaseVectorDB, SearchResult

logger = get_logger(__name__)


class DenseRetriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ):
        cfg = get_config().get("retrieval", {})
        self._embedder = embedder
        self._vectordb = vectordb
        self._top_k = top_k or cfg.get("top_k", 20)
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(cfg.get("similarity_threshold", 0.0) or 0.0)
        )

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
            results = kept
        return results
