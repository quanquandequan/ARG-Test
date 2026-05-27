"""In-memory fakes for unit/component/API tests.

These mirror the production interfaces (BaseEmbedder/BaseLLM/...) so we can
exercise the pipeline without loading real models or running Milvus.
"""

from __future__ import annotations

import numpy as np

from src.embedding.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.llm.types import ChatResponse, ContentBlock, Message
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import BaseVectorDB, SearchResult


class FakeEmbedder(BaseEmbedder):
    """Deterministic hash-based embedder. Same text → same vector."""

    def __init__(self, dim: int = 1024):
        self._dim = dim
        self._loaded = False

    def _vec_for(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for i, ch in enumerate(text):
            v[(ord(ch) * 131 + i) % self._dim] += 1.0
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._vec_for(t) for t in texts], axis=0)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec_for(query)

    def dim(self) -> int:
        return self._dim

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded


class FakeVectorDB(BaseVectorDB):
    """In-memory vector store with cosine-similarity search."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._collection_created = False

    def create_collection(
        self,
        name: str | None = None,
        dim: int | None = None,
        drop_existing: bool = False,
    ) -> None:
        if drop_existing:
            self._store.clear()
        self._collection_created = True

    def insert(self, chunks_with_vectors: list[tuple]) -> None:
        for row in chunks_with_vectors:
            chunk_id, doc_id, content, idx, vec, meta = row
            self._store[chunk_id] = {
                "id": chunk_id,
                "document_id": doc_id,
                "content": content,
                "chunk_index": idx,
                "vector": np.asarray(vec, dtype=np.float32),
                "metadata": dict(meta or {}),
            }

    def search(
        self,
        query_vector,
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        q = np.asarray(query_vector, dtype=np.float32)
        scored: list[tuple[float, dict]] = []
        for item in self._store.values():
            if filters and not self._matches(item, filters):
                continue
            score = float(np.dot(q, item["vector"]))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                id=item["id"],
                document_id=item["document_id"],
                content=item["content"],
                score=score,
                metadata=item["metadata"],
            )
            for score, item in scored[:top_k]
        ]

    @staticmethod
    def _matches(item: dict, filters: dict) -> bool:
        for key, value in filters.items():
            if key == "document_id" and item["document_id"] != value:
                return False
            meta_val = item["metadata"].get(key)
            if meta_val is not None and meta_val != value:
                return False
        return True

    def delete_by_document_id(self, document_id: str) -> int:
        ids = [cid for cid, v in self._store.items() if v["document_id"] == document_id]
        for cid in ids:
            del self._store[cid]
        return len(ids)

    def count(self) -> int:
        return len(self._store)

    def drop_collection(self) -> None:
        self._store.clear()

    def close(self) -> None:  # pragma: no cover - nothing to do
        pass


class FakeLLM(BaseLLM):
    """Configurable fake LLM for testing pipeline + Agent flows.

    Can be set to return:
    - text responses (end_turn)
    - tool calls (tool_use)
    - a sequence of mixed responses for multi-step agent tests
    """

    def __init__(
        self,
        response_text: str = "根据[1]可知答案是测试。",
        responses: list[ChatResponse] | None = None,
    ):
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_temperature: float | None = None
        self.last_messages: list[Message] | None = None
        self.last_tools: list[dict] | None = None
        self._loaded = True
        # Pre-programmed response sequence for agent testing
        self._responses = responses
        self._response_idx = 0
        self._stream_content_cache: str | None = None

    async def generate_chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.last_messages = messages
        self.last_tools = tools
        self.last_temperature = temperature

        if self._responses and self._response_idx < len(self._responses):
            resp = self._responses[self._response_idx]
            self._response_idx += 1
            self._stream_content_cache = resp.content
            return resp

        self._stream_content_cache = self.response_text
        return ChatResponse(
            content=self.response_text,
            model="fake",
            stop_reason="end_turn",
            usage={},
        )

    async def generate_chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.last_messages = messages
        self.last_tools = tools
        self.last_temperature = temperature

        if self._stream_content_cache is not None:
            content = self._stream_content_cache
            self._stream_content_cache = None
        elif self._responses and self._response_idx < len(self._responses):
            resp = self._responses[self._response_idx]
            self._response_idx += 1
            content = resp.content
        else:
            content = self.response_text

        for token in content:
            yield ContentBlock(type="text", text=token)

    def is_loaded(self) -> bool:
        return self._loaded


class FakeReranker(BaseReranker):
    """Pass-through reranker that keeps original ordering and truncates to top_k."""

    def __init__(self) -> None:
        self._loaded = True
        self.calls: int = 0

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        self.calls += 1
        n = top_k or len(candidates)
        return list(candidates[:n])

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded
