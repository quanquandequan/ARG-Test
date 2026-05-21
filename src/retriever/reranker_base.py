"""Abstract base class for rerankers."""

from abc import ABC, abstractmethod

from src.vectordb.base import SearchResult


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank candidates by relevance to query. Returns top_k results."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load/prepare the reranker."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the reranker is ready."""
        ...
