"""Component test for the Generator with all fakes wired."""

import pytest

from src.generation.generator import Generator
from src.retriever.dense_retriever import DenseRetriever


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


def _make_generator(embedder, vectordb, llm, reranker):
    retriever = DenseRetriever(embedder, vectordb)
    return Generator(
        embedder=embedder,
        vectordb=vectordb,
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )


async def test_query_returns_answer_and_citations(
    fake_embedder, fake_vectordb, fake_llm, fake_reranker
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "RAG 是检索增强生成。", 0),
            ("c2", "d2", "Milvus 是向量数据库。", 0),
        ],
    )
    fake_llm.response_text = "RAG 即检索增强生成 [1]。"
    gen = _make_generator(fake_embedder, fake_vectordb, fake_llm, fake_reranker)

    resp = await gen.query("RAG 是什么？", top_k=5, final_k=2)
    assert "RAG" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].document_id in {"d1", "d2"}


async def test_query_records_timing_stages(
    fake_embedder, fake_vectordb, fake_llm, fake_reranker
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [("c1", "d1", "内容 A", 0)],
    )
    gen = _make_generator(fake_embedder, fake_vectordb, fake_llm, fake_reranker)
    resp = await gen.query("内容")
    assert {"retrieval_ms", "rerank_ms", "llm_ms"}.issubset(resp.processing_stages.keys())
    for v in resp.processing_stages.values():
        assert v >= 0.0


async def test_empty_index_returns_no_answer_fallback(
    fake_embedder, fake_vectordb, fake_llm, fake_reranker
):
    gen = _make_generator(fake_embedder, fake_vectordb, fake_llm, fake_reranker)
    resp = await gen.query("任意查询")
    assert "无法回答" in resp.answer
    assert resp.citations == []


async def test_query_stream_yields_tokens(
    fake_embedder, fake_vectordb, fake_llm, fake_reranker
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [("c1", "d1", "流式内容", 0)],
    )
    fake_llm.response_text = "abcXY"
    gen = _make_generator(fake_embedder, fake_vectordb, fake_llm, fake_reranker)

    tokens: list[str] = []
    async for tok in gen.query_stream("流式"):
        tokens.append(tok)
    assert "".join(tokens) == "abcXY"


async def test_stream_returns_fallback_when_no_context(
    fake_embedder, fake_vectordb, fake_llm, fake_reranker
):
    gen = _make_generator(fake_embedder, fake_vectordb, fake_llm, fake_reranker)
    tokens: list[str] = []
    async for tok in gen.query_stream("无文档"):
        tokens.append(tok)
    assert "".join(tokens) == "根据现有文档无法回答此问题。"


def test_generator_requires_reranker(fake_embedder, fake_vectordb, fake_llm):
    with pytest.raises(ValueError):
        Generator(embedder=fake_embedder, vectordb=fake_vectordb, llm=fake_llm)
