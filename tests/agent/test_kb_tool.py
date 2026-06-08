"""A1：KnowledgeBaseTool 直接测试，覆盖 RAG 到 Agent 的桥接。"""

from src.agent.tools.search_kb import KnowledgeBaseTool, classify_knowledge_source
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import SearchResult


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


def _seed_with_metadata(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx, metadata in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, metadata))
    vectordb.insert(rows)


async def test_kb_tool_returns_formatted_results(fake_embedder, fake_vectordb, fake_reranker):
    """KB 工具封装检索，并格式化带来源信息的编号引用。"""
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
    """向量存储为空时，KB 工具会优雅提示。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="任意查询")
    assert "未找到" in result


async def test_kb_tool_top_k_caps_results(fake_embedder, fake_vectordb, fake_reranker):
    """top_k 会限制返回 chunks 数量（通过 reranker）。"""
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


async def test_kb_tool_groups_sources_by_priority(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    """Excel 测试用例优先，Bug 和 XMind 作为辅助来源分区展示。"""
    _seed_with_metadata(
        fake_embedder,
        fake_vectordb,
        [
            (
                "c1",
                "d1",
                "阅读器支持上一话和下一话切换",
                0,
                {"source_path": "/kb/ACN_cases.xlsx", "format": "xlsx"},
            ),
            (
                "c2",
                "d2",
                "Bug Key: ACNBUG-1 | 阅读器崩溃",
                0,
                {"source_path": "/kb/ACN_buglist.xlsx", "format": "xlsx"},
            ),
            (
                "c3",
                "d3",
                "叭嗒阅读器历史测试思路",
                0,
                {"source_path": "/kb/叭嗒阅读器.xmind", "format": "xmind"},
            ),
        ],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="叭嗒 阅读器", top_k=3)

    excel_pos = result.index("【知识库结果：Excel测试用例（事实优先）】")
    bug_pos = result.index("【知识库结果：Bug记录（辅助）】")
    xmind_pos = result.index("【知识库结果：XMind（辅助）】")
    assert excel_pos < bug_pos < xmind_pos


async def test_kb_tool_preserves_excel_candidates_before_final_cutoff(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    """即使 Excel 候选排在较后位置，也不能被最终 top_k 截断丢失。"""
    items = [
        (
            f"bug-{idx}",
            f"bug-doc-{idx}",
            f"Bug Key: ACNBUG-{idx} | 漫画阅读器历史缺陷 {idx}",
            idx,
            {"source_path": "/kb/ACN_buglist.xlsx", "format": "xlsx"},
        )
        for idx in range(20)
    ]
    items.append(
        (
            "excel-1",
            "excel-doc",
            "漫画阅读器测试用例：支持目录、上一话、下一话、关注、评论。",
            21,
            {"source_path": "/kb/ACN_cases.xlsx", "format": "xlsx"},
        )
    )
    _seed_with_metadata(fake_embedder, fake_vectordb, items)
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="漫画阅读器 功能", top_k=5)

    assert "【来源说明】已命中 Excel 测试用例" in result
    assert "【知识库结果：Excel测试用例（事实优先）】" in result
    assert "漫画阅读器测试用例" in result


async def test_kb_tool_marks_auxiliary_only_when_excel_missing(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    """无 Excel 命中时，Bug/XMind 结果必须标明仅为辅助参考。"""
    _seed_with_metadata(
        fake_embedder,
        fake_vectordb,
        [
            (
                "bug-1",
                "bug-doc",
                "Bug Key: ACNBUG-1 | 漫画阅读器崩溃",
                0,
                {"source_path": "/kb/ACN_buglist.xlsx", "format": "xlsx"},
            ),
            (
                "xmind-1",
                "xmind-doc",
                "阅读器历史测试思路：目录、设置、分享",
                1,
                {"source_path": "/kb/叭嗒阅读器.xmind", "format": "xmind"},
            ),
        ],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    result = await tool.execute(query="漫画阅读器 功能", top_k=5)

    assert "未命中 Excel 测试用例" in result
    assert "仅为辅助参考" in result


def test_classify_knowledge_source_supports_legacy_metadata():
    """来源分类兼容只有 source_path 或仅有内容特征的旧数据。"""
    assert classify_knowledge_source(SearchResult(
        id="1",
        document_id="d1",
        content="阅读器测试用例",
        score=1.0,
        metadata={"source_path": "/kb/ACN_cases.xlsx"},
    )).key == "excel_case"
    assert classify_knowledge_source(SearchResult(
        id="2",
        document_id="d2",
        content="Bug Key: ACNBUG-1 | 阅读器崩溃",
        score=1.0,
        metadata={"source_path": "/kb/unknown.xlsx"},
    )).key == "bug"
    assert classify_knowledge_source(SearchResult(
        id="3",
        document_id="d3",
        content="历史用例",
        score=1.0,
        metadata={"source_path": "/kb/叭嗒.xmind"},
    )).key == "xmind"


async def test_kb_tool_openai_schema_is_valid(fake_embedder, fake_vectordb, fake_reranker):
    """工具 schema 满足 OpenAI / Anthropic function-calling 契约。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tool = KnowledgeBaseTool(engine)

    schema = tool.to_tool_schema()
    assert schema["name"] == "knowledge_search"
    assert schema["description"]
    params = schema["parameters"]
    assert "query" in params["properties"]
    assert "query" in params["required"]
