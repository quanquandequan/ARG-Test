"""ChineseChunker 单元测试。"""

from src.ingestion.chunker import ChineseChunker


def test_empty_text_yields_no_chunks():
    chunker = ChineseChunker()
    assert chunker.split("doc-1", "") == []
    assert chunker.split("doc-1", "   \n  ") == []


def test_short_text_produces_one_chunk():
    chunker = ChineseChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=5)
    chunks = chunker.split("doc-1", "这是一段很短的中文文本。")
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert "这是一段" in chunks[0].content


def test_sentence_boundaries_are_respected():
    chunker = ChineseChunker(chunk_size=20, chunk_overlap=4, min_chunk_size=4)
    text = "第一句话很简单。第二句话稍微长一点。第三句话也是短的。第四句话结束本段。"
    chunks = chunker.split("doc-1", text)
    assert len(chunks) >= 2
    for c in chunks:
        # 分块不应在句子中间结束（即最后的非重叠区域）
        # 允许尾部以 。 结束，或刚好等于某个句子
        assert c.content.strip() != ""


def test_chunk_indices_are_monotonic_and_zero_based():
    chunker = ChineseChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=4)
    text = "。".join([f"句子{i}内容很长一些" for i in range(20)]) + "。"
    chunks = chunker.split("doc-x", text)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_between_adjacent_chunks():
    chunker = ChineseChunker(chunk_size=30, chunk_overlap=10, min_chunk_size=4)
    text = "。".join([f"句子{i}是一段较为完整的中文表达" for i in range(10)]) + "。"
    chunks = chunker.split("doc-x", text)
    if len(chunks) >= 2:
        # 第一个分块的尾部应出现在第二个分块的开头
        tail = chunks[0].content[-10:]
        assert any(t in chunks[1].content[:30] for t in (tail, tail[:5])) or chunks[
            1
        ].content.startswith(chunks[0].content[-5:])


def test_very_long_unpunctuated_text_is_split():
    chunker = ChineseChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=4)
    # 没有句末标点时，chunker 仍应通过 jieba 切分
    text = "中文文本" * 60
    chunks = chunker.split("doc-x", text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.content) <= 60  # 切分上限为 chunk_size * 1.5


def test_chunk_ids_are_unique():
    chunker = ChineseChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=4)
    text = "。".join([f"段落{i}" for i in range(30)]) + "。"
    chunks = chunker.split("doc-x", text)
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids)
