"""Reranker 抽象基类。"""

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
        """按与 query 的相关性重排候选项，返回 top_k 个结果。"""
        ...

    @abstractmethod
    def load(self) -> None:
        """加载或准备 reranker。"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查 reranker 是否就绪。"""
        ...
