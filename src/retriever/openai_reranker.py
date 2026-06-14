"""基于 OpenAI 的 reranker：使用 chat model 为每段文本相关性打分。"""

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
    """通过 OpenAI chat API 实现的 reranker，使用低成本模型（gpt-4o-mini）打分。

    为提高效率，所有候选项会在单次 API 调用中完成打分。
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

        # 用 is not None 而非 or，避免 top_k=0 被 falsy 短路而读取 config 默认值
        top_k = top_k if top_k is not None else get_config().get("retrieval", {}).get("final_k", 5)

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
        # 尝试从响应中提取 JSON
        try:
            data = json.loads(raw)
            scores = data.get("scores", [])
        except json.JSONDecodeError:
            # 尝试查找类似 JSON 的数组
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
            # 兜底返回中性分数
            return [50.0] * expected

        return [float(s) for s in scores]
