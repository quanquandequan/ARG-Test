"""RAG answer generator orchestrator."""

from dataclasses import dataclass, field

from src.core.config import get_config
from src.core.logging import get_logger
from src.embedding.base import BaseEmbedder
from src.generation.citation import Citation, CitationFormatter
from src.generation.prompt_builder import PromptBuilder
from src.llm.base import BaseLLM, LLMResponse
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import BaseVectorDB, SearchResult

logger = get_logger(__name__)


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
        self._embedder = embedder
        self._vectordb = vectordb
        self._llm = llm
        self._retriever = retriever or DenseRetriever(embedder, vectordb)
        from src.retriever.reranker_factory import get_reranker
        self._reranker = reranker or get_reranker()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._citation_formatter = citation_formatter or CitationFormatter()

    async def query(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        filters: dict | None = None,
        temperature: float | None = None,
    ) -> QueryResponse:
        cfg = get_config().get("retrieval", {})

        # Stage 1: Dense retrieval
        candidates = self._retriever.retrieve(
            query,
            top_k=top_k or cfg.get("top_k", 20),
            filters=filters,
        )
        logger.debug("retrieval_done", query=query[:50], candidates=len(candidates))

        # Stage 2: Rerank
        reranked = await self._reranker.rerank(
            query,
            candidates,
            top_k=final_k or cfg.get("final_k", 5),
        )
        logger.debug("rerank_done", query=query[:50], final=len(reranked))

        if not reranked:
            return QueryResponse(
                answer="根据现有文档无法回答此问题。",
                citations=[],
            )

        # Stage 3: Build prompts
        system_prompt = self._prompt_builder.build_system_prompt(reranked)
        user_prompt = self._prompt_builder.build_user_prompt(query)

        # Stage 4: LLM generation
        llm_response = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # Stage 5: Extract citations
        citations = self._citation_formatter.extract_citations(
            llm_response.content, reranked
        )

        return QueryResponse(
            answer=llm_response.content,
            citations=citations,
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
        cfg = get_config().get("retrieval", {})

        # Retrieval + rerank (same as non-streaming)
        candidates = self._retriever.retrieve(query, top_k=top_k or cfg.get("top_k", 20), filters=filters)
        reranked = await self._reranker.rerank(query, candidates, top_k=final_k or cfg.get("final_k", 5))

        if not reranked:
            yield "根据现有文档无法回答此问题。"
            return

        system_prompt = self._prompt_builder.build_system_prompt(reranked)
        user_prompt = self._prompt_builder.build_user_prompt(query)

        async for token in self._llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            yield token
