"""BGE-Reranker cross-encoder for precision scoring."""

import asyncio

import numpy as np

from src.core.config import get_config
from src.core.exceptions import RerankerError
from src.core.logging import get_logger
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import SearchResult

logger = get_logger(__name__)


class BgeReranker(BaseReranker):
    """Cross-encoder reranking with BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        cfg = get_config().get("reranker", {})
        self._model_name = model_name or cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
        self._device = device or cfg.get("device", "cpu")
        self._batch_size = batch_size or cfg.get("batch_size", 16)
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self._model_name,
                use_fp16=False,
                device=self._device,
            )
        except Exception as e:
            raise RerankerError(
                f"Failed to load reranker '{self._model_name}': {e}"
            ) from e

    def is_loaded(self) -> bool:
        return self._model is not None

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        if self._model is None:
            self.load()

        top_k = top_k or get_config().get("retrieval", {}).get("final_k", 5)

        def _compute():
            pairs = [(query, c.content) for c in candidates]
            scores = self._model.compute_score(pairs, batch_size=self._batch_size)
            if isinstance(scores, float):
                scores = [scores]
            return scores

        scores = await asyncio.to_thread(_compute)

        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
