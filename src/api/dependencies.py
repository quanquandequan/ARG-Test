"""Compatibility dependency facade backed by the shared application container."""

import sys
from functools import lru_cache

from src.agent.react_loop import ReActAgent
from src.application.ingestion_service import DocumentIngestionService
from src.application.requirement_services import (
    RequirementAnalysisService,
    TestCaseGenerationService,
)
from src.bootstrap import AppContainer
from src.embedding.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.retriever.reranker_base import BaseReranker
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import BaseVectorDB


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer()


def get_singleton_embedder() -> BaseEmbedder:
    return get_container().get_embedder()


def get_singleton_vectordb() -> BaseVectorDB:
    return get_container().get_vectordb()


def get_singleton_llm() -> BaseLLM:
    return get_container().get_llm()


def get_singleton_reranker() -> BaseReranker:
    return get_container().get_reranker()


def get_retrieval_engine() -> RetrievalEngine:
    return get_container().get_retrieval_engine()


def get_agent(profile_name: str = "qa_agent") -> ReActAgent:
    return get_container().get_agent(profile_name)


def get_requirement_analysis_service() -> RequirementAnalysisService:
    return get_container().get_requirement_analysis_service()


def get_test_case_generation_service() -> TestCaseGenerationService:
    return get_container().get_test_case_generation_service()


def get_document_ingestion_service() -> DocumentIngestionService:
    return get_container().get_ingestion_service()


def clear_all_caches() -> None:
    """Clear all cached singletons. Call between tests for isolation."""
    _this = sys.modules[__name__]
    for name in ("get_container",):
        fn = getattr(_this, name, None)
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
