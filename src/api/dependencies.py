"""FastAPI dependency injection — singleton components."""

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

_embedder: BaseEmbedder | None = None
_vectordb: BaseVectorDB | None = None
_llm: BaseLLM | None = None
_reranker: BaseReranker | None = None
_retrieval_engine: RetrievalEngine | None = None
_agent: ReActAgent | None = None


def get_singleton_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
        _embedder.load()
    return _embedder


def get_singleton_vectordb() -> BaseVectorDB:
    global _vectordb
    if _vectordb is None:
        _vectordb = get_vectordb()
    return _vectordb


def get_singleton_llm() -> BaseLLM:
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def get_singleton_reranker() -> BaseReranker:
    global _reranker
    if _reranker is None:
        _reranker = get_reranker()
    return _reranker


def get_retrieval_engine() -> RetrievalEngine:
    global _retrieval_engine
    if _retrieval_engine is None:
        embedder = get_singleton_embedder()
        vectordb = get_singleton_vectordb()
        reranker = get_singleton_reranker()
        dense = DenseRetriever(embedder, vectordb)
        _retrieval_engine = RetrievalEngine(
            dense_retriever=dense,
            reranker=reranker,
        )
    return _retrieval_engine


def get_agent() -> ReActAgent:
    global _agent
    if _agent is None:
        from src.core.config import get_config

        llm = get_singleton_llm()
        engine = get_retrieval_engine()
        cfg_agent = get_config().get("agent", {})
        tool_names = list(cfg_agent.get("tools", ["knowledge_search", "web_search"]))
        tools = build_agent_tools(engine, tool_names)
        _agent = ReActAgent(
            llm=llm,
            tools=tools,
            system_prompt=cfg_agent.get("system_prompt", "") or "",
            max_iterations=int(cfg_agent.get("max_iterations", 10)),
            max_history_tokens=int(cfg_agent.get("max_history_tokens", 4000)),
        )
    return _agent
