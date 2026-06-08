"""DenseRetriever 与内存 FakeVectorDB 的组件测试。"""

from src.retriever.dense_retriever import DenseRetriever


def _seed(embedder, vectordb, items):
    rows = []
    for chunk_id, doc_id, content, idx in items:
        vec = embedder.embed_query(content)
        rows.append((chunk_id, doc_id, content, idx, vec, {"source_path": f"{doc_id}.md"}))
    vectordb.insert(rows)


def test_retrieve_returns_top_k_in_score_order(fake_embedder, fake_vectordb):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "苹果是水果", 0),
            ("c2", "d2", "汽车是交通工具", 0),
            ("c3", "d3", "香蕉也是水果", 0),
        ],
    )
    retriever = DenseRetriever(fake_embedder, fake_vectordb, top_k=2)
    hits = retriever.retrieve("水果")
    assert len(hits) <= 2
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_filters_passed_through(fake_embedder, fake_vectordb):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "alpha 内容", 0),
            ("c2", "d2", "beta 内容", 0),
        ],
    )
    retriever = DenseRetriever(fake_embedder, fake_vectordb, top_k=10)
    hits = retriever.retrieve("内容", filters={"document_id": "d2"})
    assert {h.document_id for h in hits} == {"d2"}


def test_similarity_threshold_filters_low_scores(fake_embedder, fake_vectordb):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "同一个查询字符串", 0),
            ("c2", "d2", "完全不相关的另一段内容", 0),
        ],
    )
    retriever = DenseRetriever(
        fake_embedder, fake_vectordb, top_k=10, similarity_threshold=0.99
    )
    hits = retriever.retrieve("同一个查询字符串")
    # 只有（近似）精确匹配能通过非常严格的阈值
    assert all(h.score >= 0.99 for h in hits)
    assert any(h.document_id == "d1" for h in hits)


def test_threshold_zero_keeps_everything(fake_embedder, fake_vectordb):
    _seed(
        fake_embedder,
        fake_vectordb,
        [("c1", "d1", "X", 0), ("c2", "d2", "Y", 0)],
    )
    retriever = DenseRetriever(
        fake_embedder, fake_vectordb, top_k=10, similarity_threshold=0.0
    )
    hits = retriever.retrieve("Z")
    assert len(hits) == 2


def test_threshold_keeps_candidates_when_all_would_be_dropped(
    fake_embedder,
    fake_vectordb,
):
    _seed(
        fake_embedder,
        fake_vectordb,
        [
            ("c1", "d1", "叭嗒漫画 阅读器 功能介绍", 0),
            ("c2", "d2", "叭嗒漫画 阅读设置 离线下载", 0),
        ],
    )
    retriever = DenseRetriever(
        fake_embedder,
        fake_vectordb,
        top_k=10,
        similarity_threshold=1.01,
    )

    hits = retriever.retrieve("叭嗒漫画阅读器都有哪些功能")

    assert len(hits) == 2
