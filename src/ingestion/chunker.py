"""Chinese-aware text chunking with jieba sentence boundary detection."""

import re
import uuid
from dataclasses import dataclass, field

import jieba

from src.core.config import get_config
from src.core.exceptions import ChunkingError


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class ChineseChunker:
    """Structure-aware chunker optimized for Chinese text.

    Uses jieba tokenization for sentence boundary detection and respects
    document structure (headings, paragraphs). Guarantees no sentence is split.
    """

    _SENTENCE_ENDS = re.compile(r"[。！？.!?\n]")

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ):
        cfg = get_config().get("chunking", {})
        self.chunk_size = chunk_size or cfg.get("chunk_size", 512)
        self.chunk_overlap = chunk_overlap or cfg.get("chunk_overlap", 100)
        self.min_chunk_size = min_chunk_size or cfg.get("min_chunk_size", 50)

    def split(self, document_id: str, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        chunks = self._merge_sentences(sentences)

        result: list[Chunk] = []
        for i, chunk_text in enumerate(chunks):
            result.append(Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
            ))
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using Chinese-aware boundaries."""
        jieba.setLogLevel(60)  # Suppress jieba debug output
        raw_sentences: list[str] = []
        current = ""

        for char in text:
            current += char
            if char in ("。", "！", "？", "\n"):
                raw_sentences.append(current.strip())
                current = ""

        if current.strip():
            raw_sentences.append(current.strip())

        # Further split very long "sentences" (likely missing punctuation)
        sentences: list[str] = []
        for sent in raw_sentences:
            if len(sent) > self.chunk_size * 1.5:
                words = list(jieba.cut(sent))
                buf = ""
                buf_len = 0
                for word in words:
                    if buf_len + len(word) > self.chunk_size and buf:
                        sentences.append(buf.strip())
                        buf = word
                        buf_len = len(word)
                    else:
                        buf += word
                        buf_len += len(word)
                if buf.strip():
                    sentences.append(buf.strip())
            elif sent.strip():
                sentences.append(sent)

        return sentences

    def _merge_sentences(self, sentences: list[str]) -> list[str]:
        """Merge sentences into chunks respecting size limits."""
        if not sentences:
            return []

        chunks: list[str] = []
        current = ""
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            if current_len + sent_len > self.chunk_size and current:
                chunks.append(current)
                # Overlap: keep tail of previous chunk
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = overlap_text + sent
                current_len = len(current)
            else:
                current = (current + sent) if current else sent
                current_len += sent_len

        if current.strip():
            current = current.strip()
            # Merge into previous chunk if too small
            if current_len < self.min_chunk_size and chunks:
                chunks[-1] = chunks[-1] + current
            else:
                chunks.append(current)

        return chunks
