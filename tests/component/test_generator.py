"""Component test for the Generator retrieval engine."""

import pytest

from src.generation.generator import Generator
from src.retriever.dense_retriever import DenseRetriever


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


def _make_generator(embedder, vectordb, reranker):
    retriever = DenseRetriever(embedder, vectordb)
    return Generator(embedder=embedder, vectordb=vectordb, retriever=retriever, reranker=reranker)


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
    gen = _make_generator(fake_embedder, fake_vectordb, fake_reranker)

    results = await gen.search("RAG 是什么？", top_k=5, final_k=2)
    assert len(results) >= 1
    assert isinstance(results[0].content, str)


async def test_search_empty_index_returns_empty(
    fake_embedder, fake_vectordb, fake_reranker
):
    gen = _make_generator(fake_embedder, fake_vectordb, fake_reranker)
    results = await gen.search("任意查询")
    assert results == []


def test_generator_requires_reranker(fake_embedder, fake_vectordb):
    with pytest.raises(ValueError):
        Generator(embedder=fake_embedder, vectordb=fake_vectordb)
