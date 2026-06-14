"""检索引擎：retrieve -> rerank -> chunks。"""

import asyncio

from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import SearchResult


class RetrievalEngine:
    """从知识库检索文档并重排。

    暴露单一 ``search()`` 方法，运行完整检索链路并返回排序后的 chunks。
    LLM 生成由 Agent 层处理；本类只返回 ``SearchResult`` 对象。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: BaseReranker,
        top_k: int = 20,
        final_k: int = 5,
    ):
        if reranker is None:
            raise ValueError(
                "RetrievalEngine requires an explicit reranker; "
                "build one via reranker_factory.get_reranker() and inject."
            )
        self._dense_retriever = dense_retriever
        self._reranker = reranker
        self._top_k = top_k
        self._final_k = final_k

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """运行 retrieve -> rerank 并返回排序后的 chunks。"""
        resolved_top_k = top_k if top_k is not None else self._top_k
        resolved_final_k = final_k if final_k is not None else self._final_k
        candidates = await self.retrieve_candidates(
            query=query,
            top_k=resolved_top_k,
            filters=filters,
        )
        if not candidates:
            return []
        return await self.rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=resolved_final_k,
        )

    async def retrieve_candidates(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """只运行 dense retrieve，返回未重排候选。"""
        # 将同步 retrieve（embedding + 向量搜索）放入 to_thread，
        # 避免阻塞异步事件循环。
        return await asyncio.to_thread(
            self._dense_retriever.retrieve,
            query,
            top_k,
            filters,
        )

    async def rerank_candidates(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """对指定候选运行 rerank，便于工具层构造来源感知候选池。"""
        if not candidates:
            return []
        return await self._reranker.rerank(query, candidates, top_k=top_k)
