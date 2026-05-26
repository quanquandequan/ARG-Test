"""Unit tests for CitationFormatter."""

from src.generation.citation import CitationFormatter
from src.vectordb.base import SearchResult


def _make_chunks(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            id=f"c{i}",
            document_id=f"doc{i}",
            content=f"内容{i}" * 30,
            score=1.0 - i * 0.1,
            metadata={"source_path": f"/tmp/file{i}.md", "chunk_index": i},
        )
        for i in range(n)
    ]


def test_no_citation_markers_returns_empty():
    formatter = CitationFormatter()
    cits = formatter.extract_citations("这是一段没有引用的答案。", _make_chunks(3))
    assert cits == []


def test_extracts_single_citation():
    formatter = CitationFormatter()
    chunks = _make_chunks(3)
    cits = formatter.extract_citations("答案见[2]。", chunks)
    assert len(cits) == 1
    assert cits[0].document_id == "doc1"  # [2] → index 1
    assert cits[0].chunk_index == 1


def test_extracts_multiple_in_sorted_unique_order():
    formatter = CitationFormatter()
    chunks = _make_chunks(5)
    cits = formatter.extract_citations("先[3]再[1]最后[1]又见[2]", chunks)
    # Deduped, sorted ascending
    assert [c.document_id for c in cits] == ["doc0", "doc1", "doc2"]


def test_out_of_range_marker_is_dropped():
    formatter = CitationFormatter()
    chunks = _make_chunks(2)
    cits = formatter.extract_citations("看[9]或[1]", chunks)
    assert [c.document_id for c in cits] == ["doc0"]


def test_text_field_truncated_to_200_chars():
    formatter = CitationFormatter()
    chunks = _make_chunks(1)
    cits = formatter.extract_citations("见[1]", chunks)
    assert len(cits) == 1
    assert len(cits[0].text) <= 200


def test_metadata_fields_populated():
    formatter = CitationFormatter()
    chunks = _make_chunks(1)
    cits = formatter.extract_citations("见[1]", chunks)
    assert cits[0].source_path == "/tmp/file0.md"
    assert cits[0].chunk_index == 0
    assert cits[0].relevance_score == chunks[0].score
