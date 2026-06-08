"""RetrievalEngine 组件测试。"""

import pytest

from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


def _make_engine(embedder, vectordb, reranker):
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


async def test_search_returns_ranked_results(
    fake_embedder, fake_vectordb, fake_reranker
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "RAG 是检索增强生成。", 0),
            ("c2", "d2", "Milvus 是向量数据库。", 0),
        ],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)

    results = await engine.search("RAG 是什么？", top_k=5, final_k=2)
    assert len(results) >= 1
    assert isinstance(results[0].content, str)


async def test_search_empty_index_returns_empty(
    fake_embedder, fake_vectordb, fake_reranker
):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    results = await engine.search("任意查询")
    assert results == []


async def test_retrieve_and_rerank_candidates_are_reusable(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "阅读器目录功能", 0),
            ("c2", "d2", "阅读器评论功能", 0),
        ],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)

    candidates = await engine.retrieve_candidates("阅读器", top_k=5)
    ranked = await engine.rerank_candidates("阅读器", candidates, top_k=1)

    assert len(candidates) == 2
    assert len(ranked) == 1
    assert fake_reranker.calls == 1


async def test_search_keeps_existing_behavior(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "阅读器目录功能", 0),
            ("c2", "d2", "阅读器评论功能", 0),
        ],
    )
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)

    results = await engine.search("阅读器", top_k=5, final_k=1)

    assert len(results) == 1
    assert fake_reranker.calls == 1


def test_retrieval_engine_requires_reranker(fake_embedder, fake_vectordb):
    dense = DenseRetriever(fake_embedder, fake_vectordb)
    with pytest.raises((ValueError, TypeError)):
        RetrievalEngine(dense_retriever=dense, reranker=None)
