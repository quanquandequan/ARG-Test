"""Retrieval engine — retrieve -> rerank -> chunks."""

import asyncio

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
        top_k: int = 20,
        final_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run retrieve -> rerank and return ranked chunks."""
        # Wrap synchronous retrieve (embedding + vector search) in to_thread
        # so it does not block the async event loop.
        candidates = await asyncio.to_thread(
            self._dense_retriever.retrieve,
            query,
            top_k,
            filters,
        )
        if not candidates:
            return []
        return await self._reranker.rerank(query, candidates, top_k=final_k)
