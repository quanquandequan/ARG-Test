"""OpenAI-based reranker — uses a chat model to score relevance of each passage."""

import json
import os
import re

from src.core.config import get_config
from src.core.exceptions import RerankerError
from src.core.logging import get_logger
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import SearchResult

logger = get_logger(__name__)

_RERANK_PROMPT = """Score each passage's relevance to the query on a scale of 0-100.
Return ONLY a JSON object with a "scores" key containing an array of integers.

Query: {query}

Passages:
{passages}

JSON:"""


class OpenAIReranker(BaseReranker):
    """Reranker via OpenAI chat API. Uses a cheap model (gpt-4o-mini) for scoring.

    All candidates are scored in a single API call for efficiency.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        cfg = get_config().get("reranker", {})
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or cfg.get("model", "gpt-4o-mini")
        self._client = None

    def load(self) -> None:
        if not self._api_key:
            raise RerankerError("OPENAI_API_KEY not set for OpenAI reranker")

    def is_loaded(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        top_k = top_k or get_config().get("retrieval", {}).get("final_k", 5)

        passages_block = "\n\n".join(
            f"[{i}] {c.content[:800]}" for i, c in enumerate(candidates)
        )
        prompt = _RERANK_PROMPT.format(query=query, passages=passages_block)

        try:
            response = await self._get_client().chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            raise RerankerError(f"OpenAI reranker API error: {e}") from e

        scores = self._parse_scores(raw, len(candidates))

        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    def _parse_scores(self, raw: str, expected: int) -> list[float]:
        # Try to extract JSON from the response
        try:
            data = json.loads(raw)
            scores = data.get("scores", [])
        except json.JSONDecodeError:
            # Try to find a JSON-like array
            match = re.search(r"\[[\d,\s]+\]", raw)
            if match:
                try:
                    scores = json.loads(match.group())
                except json.JSONDecodeError:
                    scores = []
            else:
                scores = []

        if not scores or len(scores) != expected:
            logger.warning(
                "reranker_score_mismatch",
                expected=expected,
                got=len(scores),
                raw=raw[:200],
            )
            # Fallback: return neutral scores
            return [50.0] * expected

        return [float(s) for s in scores]
