"""FastAPI dependency injection — cached singleton components."""

import sys
from functools import lru_cache

from src.agent.react_loop import ReActAgent
from src.agent.tool_factory import build_agent_tools
from src.embedding.base import BaseEmbedder
from src.embedding.factory import get_embedder
from src.llm.base import BaseLLM
from src.llm.factory import get_llm
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.retriever.reranker_factory import get_reranker
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import BaseVectorDB
from src.vectordb.factory import get_vectordb


@lru_cache(maxsize=1)
def get_singleton_embedder() -> BaseEmbedder:
    embedder = get_embedder()
    embedder.load()
    return embedder


@lru_cache(maxsize=1)
def get_singleton_vectordb() -> BaseVectorDB:
    return get_vectordb()


@lru_cache(maxsize=1)
def get_singleton_llm() -> BaseLLM:
    return get_llm()


@lru_cache(maxsize=1)
def get_singleton_reranker() -> BaseReranker:
    return get_reranker()


@lru_cache(maxsize=1)
def get_retrieval_engine() -> RetrievalEngine:
    embedder = get_singleton_embedder()
    vectordb = get_singleton_vectordb()
    reranker = get_singleton_reranker()
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


@lru_cache(maxsize=1)
def get_agent() -> ReActAgent:
    from src.core.config import get_config

    llm = get_singleton_llm()
    engine = get_retrieval_engine()
    cfg_agent = get_config().get("agent", {})
    tool_names = list(cfg_agent.get("tools", ["knowledge_search", "web_search"]))
    tools = build_agent_tools(engine, tool_names)
    return ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=cfg_agent.get("system_prompt", "") or "",
        max_iterations=int(cfg_agent.get("max_iterations", 10)),
        max_history_tokens=int(cfg_agent.get("max_history_tokens", 4000)),
    )


def clear_all_caches() -> None:
    """Clear all cached singletons. Call between tests for isolation.

    Uses getattr to safely handle monkeypatched functions that may not
    have a cache_clear method.
    """
    _this = sys.modules[__name__]
    for name in (
        "get_singleton_embedder",
        "get_singleton_vectordb",
        "get_singleton_llm",
        "get_singleton_reranker",
        "get_retrieval_engine",
        "get_agent",
    ):
        fn = getattr(_this, name, None)
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
