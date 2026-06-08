"""SearchKnowledgeTool 测试。"""

from __future__ import annotations

from src.agent.tools.search_knowledge import SearchKnowledgeTool
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine


class _FakeWebTool:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    async def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.response


class _NoHitKnowledgeTool:
    async def search_typed(self, **kwargs):
        from src.agent.tools.search_kb import KnowledgeSearchResult

        return KnowledgeSearchResult(
            content="知识库无匹配。",
            hit_count=0,
            results=[],
        )


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


async def test_search_knowledge_prefers_kb(fake_embedder, fake_vectordb, fake_reranker):
    _seed(
        fake_embedder,
        fake_vectordb,
        [("c1", "d1", "登录功能支持账号密码登录", 0)],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    web_tool = _FakeWebTool("网页结果")
    tool = SearchKnowledgeTool(engine, web_tool=web_tool)

    result = await tool.execute(query="登录功能", need_fresh_info=False)

    assert "【知识库结果】" in result
    assert "账号密码登录" in result
    assert "【网页结果】" not in result
    assert web_tool.calls == []


async def test_search_knowledge_does_not_use_web_when_kb_hits_even_if_fresh_requested(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [("c1", "d1", "阅读器支持上一话和下一话切换", 0)],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    web_tool = _FakeWebTool("网页结果：离线下载")
    tool = SearchKnowledgeTool(engine, web_tool=web_tool)

    result = await tool.execute(
        query="叭嗒漫画阅读器功能",
        need_fresh_info=True,
    )

    assert "上一话和下一话" in result
    assert "离线下载" not in result
    assert "【网页结果" not in result
    assert web_tool.calls == []


async def test_search_knowledge_falls_back_to_web(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    web_tool = _FakeWebTool("网页搜索命中：最新发布说明")
    tool = SearchKnowledgeTool(engine, web_tool=web_tool)

    result = await tool.execute(query="最新发布说明", need_fresh_info=False)

    assert "【知识库结果】" in result
    assert "未找到相关文档" in result
    assert "【网页结果（知识库无命中时补充）】" in result
    assert "最新发布说明" in web_tool.calls[0]["query"]


async def test_search_knowledge_fallback_uses_structured_hit_count(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    web_tool = _FakeWebTool("网页兜底结果")
    tool = SearchKnowledgeTool(engine, web_tool=web_tool)
    tool._kb_tool = _NoHitKnowledgeTool()

    result = await tool.execute(query="任意问题")

    assert "知识库无匹配。" in result
    assert "【网页结果（知识库无命中时补充）】" in result
    assert web_tool.calls == [
        {"query": "任意问题", "num_results": 5}
    ]
