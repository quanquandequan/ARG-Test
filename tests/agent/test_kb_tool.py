"""A1: Direct tests for KnowledgeBaseTool — the RAG-to-Agent bridge."""

from src.agent.tools.search_kb import KnowledgeBaseTool
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


async def test_kb_tool_returns_formatted_results(fake_embedder, fake_vectordb, fake_reranker):
    """KB tool wraps retrieval and formats numbered citations with source info."""
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "RAG 是检索增强生成技术", 0),
            ("c2", "d2", "Milvus 是一个向量数据库", 0),
        ],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="RAG", top_k=3)

    assert isinstance(result, str)
    assert "找到" in result
    assert "[1]" in result
    assert "来源" in result


async def test_kb_tool_empty_index_returns_not_found(fake_embedder, fake_vectordb, fake_reranker):
    """KB tool gracefully reports when vector store is empty."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="任意查询")
    assert "未找到" in result


async def test_kb_tool_top_k_caps_results(fake_embedder, fake_vectordb, fake_reranker):
    """top_k limits the number of returned chunks (via reranker)."""
    _seed(
        fake_embedder,
        fake_vectordb,
        [(f"c{i}", f"d{i}", f"内容片段{i}", 0) for i in range(5)],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="内容", top_k=2)
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" not in result


async def test_kb_tool_openai_schema_is_valid(fake_embedder, fake_vectordb, fake_reranker):
    """Tool schema satisfies OpenAI / Anthropic function-calling contract."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    schema = tool.to_tool_schema()
    assert schema["name"] == "knowledge_search"
    assert schema["description"]
    params = schema["parameters"]
    assert "query" in params["properties"]
    assert "query" in params["required"]
