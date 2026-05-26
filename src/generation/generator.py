"""RAG answer generator orchestrator."""

import time
from dataclasses import dataclass, field

from src.core.config import get_config
from src.core.logging import get_logger
from src.embedding.base import BaseEmbedder
from src.generation.citation import Citation, CitationFormatter
from src.generation.prompt_builder import PromptBuilder
from src.llm.base import BaseLLM
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import BaseVectorDB, SearchResult

logger = get_logger(__name__)

_NO_ANSWER = "根据现有文档无法回答此问题。"


@dataclass
class QueryResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    processing_stages: dict[str, float] = field(default_factory=dict)


class Generator:
    """Orchestrate the full RAG query pipeline: retrieve → rerank → generate → cite."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        llm: BaseLLM,
        retriever: DenseRetriever | None = None,
        reranker: BaseReranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_formatter: CitationFormatter | None = None,
    ):
        if reranker is None:
            raise ValueError(
                "Generator requires an explicit reranker; "
                "build one via reranker_factory.get_reranker() and inject."
            )
        self._embedder = embedder
        self._vectordb = vectordb
        self._llm = llm
        self._retriever = retriever or DenseRetriever(embedder, vectordb)
        self._reranker = reranker
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._citation_formatter = citation_formatter or CitationFormatter()

    async def _retrieve_and_rerank(
        self,
        query: str,
        top_k: int | None,
        final_k: int | None,
        filters: dict | None,
        stages: dict[str, float],
    ) -> list[SearchResult]:
        cfg = get_config().get("retrieval", {})

        t0 = time.perf_counter()
        candidates = self._retriever.retrieve(
            query,
            top_k=top_k or cfg.get("top_k", 20),
            filters=filters,
        )
        stages["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "retrieval_done",
            query=query[:50],
            candidates=len(candidates),
            duration_ms=stages["retrieval_ms"],
        )

        if not candidates:
            stages["rerank_ms"] = 0.0
            return []

        t1 = time.perf_counter()
        reranked = await self._reranker.rerank(
            query,
            candidates,
            top_k=final_k or cfg.get("final_k", 5),
        )
        stages["rerank_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        logger.debug(
            "rerank_done",
            query=query[:50],
            final=len(reranked),
            duration_ms=stages["rerank_ms"],
        )
        return reranked

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run retrieve → rerank and return ranked chunks (no LLM generation)."""
        cfg = get_config().get("retrieval", {})
        stages: dict[str, float] = {}
        return await self._retrieve_and_rerank(
            query, top_k or cfg.get("top_k", 20), final_k or cfg.get("final_k", 5), filters, stages
        )

    async def query(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
        temperature: float | None = None,
    ) -> QueryResponse:
        stages: dict[str, float] = {}
        reranked = await self._retrieve_and_rerank(
            query, top_k, final_k, filters, stages
        )

        if not reranked:
            return QueryResponse(
                answer=_NO_ANSWER,
                citations=[],
                processing_stages=stages,
            )

        system_prompt = self._prompt_builder.build_system_prompt(reranked)
        user_prompt = self._prompt_builder.build_user_prompt(query)

        t = time.perf_counter()
        llm_response = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        stages["llm_ms"] = round((time.perf_counter() - t) * 1000, 2)

        citations = self._citation_formatter.extract_citations(
            llm_response.content, reranked
        )

        return QueryResponse(
            answer=llm_response.content,
            citations=citations,
            processing_stages=stages,
        )

    async def query_stream(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
        temperature: float | None = None,
    ):
        """Stream the answer token by token."""
        stages: dict[str, float] = {}
        reranked = await self._retrieve_and_rerank(
            query, top_k, final_k, filters, stages
        )

        if not reranked:
            yield _NO_ANSWER
            return

        system_prompt = self._prompt_builder.build_system_prompt(reranked)
        user_prompt = self._prompt_builder.build_user_prompt(query)

        async for token in self._llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            yield token
