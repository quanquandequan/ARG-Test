"""Dense vector retriever — embed query → search Milvus."""

from src.core.config import get_config
from src.embedding.base import BaseEmbedder
from src.vectordb.base import BaseVectorDB, SearchResult


class DenseRetriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        top_k: int | None = None,
    ):
        self._embedder = embedder
        self._vectordb = vectordb
        self._top_k = top_k or get_config().get("retrieval", {}).get("top_k", 20)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_vec = self._embedder.embed_query(query)
        results = self._vectordb.search(query_vec, top_k=top_k or self._top_k, filters=filters)
        return results
