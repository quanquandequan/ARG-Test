"""A1: Direct tests for KnowledgeBaseTool — the RAG-to-Agent bridge."""

from src.agent.tools.search_kb import KnowledgeBaseTool
from src.generation.generator import Generator
from src.retriever.dense_retriever import DenseRetriever


def _make_generator(embedder, vectordb, reranker) -> Generator:
    retriever = DenseRetriever(embedder, vectordb)
    return Generator(embedder=embedder, vectordb=vectordb, retriever=retriever, reranker=reranker)


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
    gen = _make_generator(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(gen)

    result = await tool.execute(query="RAG", top_k=3)

    assert isinstance(result, str)
    assert "找到" in result
    assert "[1]" in result
    assert "来源" in result


async def test_kb_tool_empty_index_returns_not_found(fake_embedder, fake_vectordb, fake_reranker):
    """KB tool gracefully reports when vector store is empty."""
    gen = _make_generator(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(gen)

    result = await tool.execute(query="任意查询")
    assert "未找到" in result


async def test_kb_tool_top_k_caps_results(fake_embedder, fake_vectordb, fake_reranker):
    """top_k limits the number of returned chunks (via reranker)."""
    _seed(
        fake_embedder,
        fake_vectordb,
        [(f"c{i}", f"d{i}", f"内容片段{i}", 0) for i in range(5)],
    )
    gen = _make_generator(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(gen)

    result = await tool.execute(query="内容", top_k=2)
    # FakeReranker preserves order and truncates to top_k=2
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" not in result


async def test_kb_tool_openai_schema_is_valid(fake_embedder, fake_vectordb, fake_reranker):
    """Tool schema satisfies OpenAI / Anthropic function-calling contract."""
    gen = _make_generator(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(gen)

    schema = tool.to_openai_schema()
    assert schema["name"] == "knowledge_search"
    assert schema["description"]
    params = schema["parameters"]
    assert "query" in params["properties"]
    assert "query" in params["required"]
