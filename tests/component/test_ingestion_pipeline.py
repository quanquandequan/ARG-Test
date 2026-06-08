"""load → clean → chunk 流程的组件测试。"""

from pathlib import Path

import pytest

from src.core.exceptions import IngestionError
from src.ingestion.pipeline import IngestionPipeline


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_ingest_markdown_produces_chunks(tmp_path: Path):
    md = "# 标题\n\n这是第一段。\n\n这是第二段，内容稍长一些。"
    path = _write(tmp_path, "demo.md", md)

    pipeline = IngestionPipeline()
    doc, chunks = pipeline.ingest(path)

    assert doc.content  # 已清洗内容
    assert len(chunks) >= 1
    assert all(c.document_id == doc.id for c in chunks)
    assert all(c.content.strip() for c in chunks)


def test_ingest_text_file(tmp_path: Path):
    raw = "第一句话。第二句话。第三句话。"
    path = _write(tmp_path, "demo.txt", raw)

    pipeline = IngestionPipeline()
    doc, chunks = pipeline.ingest(path)
    assert chunks
    joined = "".join(c.content for c in chunks)
    assert "第一句话" in joined
    assert "第三句话" in joined


def test_ingest_unsupported_extension_raises(tmp_path: Path):
    path = _write(tmp_path, "demo.unknown", "content")
    pipeline = IngestionPipeline()
    with pytest.raises(IngestionError):
        pipeline.ingest(path)


def test_ingest_missing_file_raises(tmp_path: Path):
    pipeline = IngestionPipeline()
    with pytest.raises(IngestionError):
        pipeline.ingest(tmp_path / "does_not_exist.md")


def test_ingest_and_store_persists_document_metadata(fake_embedder, fake_vectordb, tmp_path: Path):
    path = _write(tmp_path, "cases.md", "# 标题\n\n测试内容。")
    pipeline = IngestionPipeline(
        embedder=fake_embedder,
        vectordb=fake_vectordb,
    )

    result = pipeline.ingest_and_store(path)

    assert result.chunks
    rows = list(fake_vectordb._store.values())
    assert rows[0]["metadata"]["source_name"] == "cases.md"
    assert rows[0]["metadata"]["source_ext"] == ".md"
    assert rows[0]["metadata"]["source_format"] == "md"
