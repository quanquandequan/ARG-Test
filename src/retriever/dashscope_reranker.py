"""DashScope (阿里云百炼) Reranker — qwen3-rerank model."""

import os

import httpx

from src.core.config import get_config
from src.core.exceptions import RerankerError
from src.core.logging import get_logger
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import SearchResult

logger = get_logger(__name__)

_DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
_DEFAULT_TIMEOUT = 30


class DashScopeReranker(BaseReranker):
    """Reranker via DashScope qwen3-rerank API.

    Uses a per-request ``httpx.AsyncClient`` context manager so the underlying
    HTTP connection is always properly closed, avoiding the "Unclosed client"
    warning that a long-lived singleton client would produce on process exit.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        cfg = get_config().get("reranker", {})
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._model = model or cfg.get("model", "qwen3-rerank")
        self._timeout = timeout

    def load(self) -> None:
        if not self._api_key:
            raise RerankerError("DASHSCOPE_API_KEY not set for DashScope reranker")

    def is_loaded(self) -> bool:
        return bool(self._api_key)

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        top_k = top_k or get_config().get("retrieval", {}).get("final_k", 5)
        documents = [c.content[:800] for c in candidates]

        body = {
            "model": self._model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": min(top_k, len(documents)),
                "return_documents": False,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    _DASHSCOPE_RERANK_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise RerankerError(f"DashScope reranker API error: {e}") from e

        output = data.get("output", {})
        results = output.get("results", [])

        if not results:
            logger.warning("reranker_empty_results", query=query[:50])
            return candidates[:top_k]

        score_map: dict[int, float] = {}
        for r in results:
            score_map[r["index"]] = r.get("relevance_score", 0.0)

        for i, candidate in enumerate(candidates):
            candidate.score = score_map.get(i, 0.0)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
