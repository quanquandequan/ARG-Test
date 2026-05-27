"""RAG retrieval engine — embed → retrieve → rerank."""

from src.core.config import get_config
from src.embedding.base import BaseEmbedder
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import BaseVectorDB, SearchResult


class Generator:
    """Thin retrieval-only wrapper: embed → retrieve → rerank → chunks.

    LLM generation is handled by the Agent layer; this class only returns
    ranked SearchResult objects for use as tool output.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        retriever: DenseRetriever | None = None,
        reranker: BaseReranker | None = None,
    ):
        if reranker is None:
            raise ValueError(
                "Generator requires an explicit reranker; "
                "build one via reranker_factory.get_reranker() and inject."
            )
        self._retriever = retriever or DenseRetriever(embedder, vectordb)
        self._reranker = reranker

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run retrieve → rerank and return ranked chunks (no LLM generation)."""
        cfg = get_config().get("retrieval", {})
        actual_top_k = top_k or cfg.get("top_k", 20)
        actual_final_k = final_k or cfg.get("final_k", 5)

        candidates = self._retriever.retrieve(query, top_k=actual_top_k, filters=filters)
        if not candidates:
            return []
        return await self._reranker.rerank(query, candidates, top_k=actual_final_k)
