"""Tests for config-driven agent tool factory."""

from src.agent.tool_factory import build_agent_tools
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


def test_build_agent_tools_default_names(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine)
    names = [t.name for t in tools]
    assert names == ["knowledge_search", "web_search"]


def test_build_agent_tools_respects_config_order(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["web_search", "knowledge_search"])
    assert [t.name for t in tools] == ["web_search", "knowledge_search"]


def test_build_agent_tools_skips_unknown_names(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["knowledge_search", "ghost_tool"])
    assert [t.name for t in tools] == ["knowledge_search"]
