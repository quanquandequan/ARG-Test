"""FastAPI dependency injection — singleton components."""

from src.agent.react_loop import ReActAgent
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.embedding.base import BaseEmbedder
from src.embedding.factory import get_embedder
from src.generation.generator import Generator
from src.llm.base import BaseLLM
from src.llm.factory import get_llm
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.retriever.reranker_factory import get_reranker
from src.vectordb.base import BaseVectorDB
from src.vectordb.factory import get_vectordb

_embedder: BaseEmbedder | None = None
_vectordb: BaseVectorDB | None = None
_llm: BaseLLM | None = None
_reranker: BaseReranker | None = None
_generator: Generator | None = None
_agent: ReActAgent | None = None


def _singleton_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
        _embedder.load()
    return _embedder


def _singleton_vectordb() -> BaseVectorDB:
    global _vectordb
    if _vectordb is None:
        _vectordb = get_vectordb()
    return _vectordb


def _singleton_llm() -> BaseLLM:
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def _singleton_reranker() -> BaseReranker:
    global _reranker
    if _reranker is None:
        _reranker = get_reranker()
    return _reranker


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        embedder = _singleton_embedder()
        vectordb = _singleton_vectordb()
        reranker = _singleton_reranker()
        retriever = DenseRetriever(embedder, vectordb)
        _generator = Generator(
            embedder=embedder,
            vectordb=vectordb,
            retriever=retriever,
            reranker=reranker,
        )
    return _generator


def get_agent() -> ReActAgent:
    global _agent
    if _agent is None:
        from src.core.config import get_config
        llm = _singleton_llm()
        generator = get_generator()
        cfg_agent = get_config().get("agent", {})
        tools = [
            KnowledgeBaseTool(generator),
            WebSearchTool(),
        ]
        _agent = ReActAgent(
            llm=llm,
            tools=tools,
            system_prompt=cfg_agent.get("system_prompt", "") or "",
            max_iterations=int(cfg_agent.get("max_iterations", 10)),
        )
    return _agent
