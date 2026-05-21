"""Citation extraction and formatting."""

import re
from dataclasses import dataclass

from src.vectordb.base import SearchResult


@dataclass
class Citation:
    text: str
    document_id: str
    source_path: str
    chunk_index: int
    relevance_score: float


class CitationFormatter:
    """Parse LLM output for citation markers and map them to source metadata."""

    _CITE_PATTERN = re.compile(r"\[(\d+)\]")

    def extract_citations(
        self,
        answer: str,
        context_chunks: list[SearchResult],
    ) -> list[Citation]:
        """Find citation markers in answer and return matching source citations."""
        seen: set[int] = set()
        for match in self._CITE_PATTERN.finditer(answer):
            index = int(match.group(1))
            seen.add(index)

        citations: list[Citation] = []
        for idx in sorted(seen):
            if 1 <= idx <= len(context_chunks):
                chunk = context_chunks[idx - 1]
                citations.append(Citation(
                    text=chunk.content[:200],
                    document_id=chunk.document_id,
                    source_path=chunk.metadata.get("source_path", ""),
                    chunk_index=chunk.metadata.get("chunk_index", 0),
                    relevance_score=chunk.score,
                ))

        return citations
