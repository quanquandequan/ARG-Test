"""Retrieval engine — retrieve → rerank → chunks."""

from src.core.config import get_config
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import SearchResult


class RetrievalEngine:
    """Retrieve and rerank documents from the knowledge base.

    Exposes a single ``search()`` method that runs the full retrieval
    pipeline and returns ranked chunks.  LLM generation is handled by
    the Agent layer; this class only returns ``SearchResult`` objects.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: BaseReranker,
    ):
        if reranker is None:
            raise ValueError(
                "RetrievalEngine requires an explicit reranker; "
                "build one via reranker_factory.get_reranker() and inject."
            )
        self._dense_retriever = dense_retriever
        self._reranker = reranker

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run retrieve → rerank and return ranked chunks."""
        cfg = get_config().get("retrieval", {})
        actual_top_k = top_k or cfg.get("top_k", 20)
        actual_final_k = final_k or cfg.get("final_k", 5)

        candidates = self._dense_retriever.retrieve(
            query, top_k=actual_top_k, filters=filters
        )
        if not candidates:
            return []
        return await self._reranker.rerank(query, candidates, top_k=actual_final_k)
