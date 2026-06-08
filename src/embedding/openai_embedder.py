"""OpenAI 兼容的 Embedding API provider。

可用于 OpenAI、DashScope（千问）以及任意 OpenAI 兼容端点。
"""

import os

import numpy as np

from src.core.config import get_config
from src.core.exceptions import EmbeddingError
from src.embedding.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """通过 API 调用 OpenAI 兼容 embedding。

    OpenAI: text-embedding-3-small (1536d) / text-embedding-3-large (3072d)
    DashScope: text-embedding-v4 (1024d, configurable)
    """

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-v4": 1024,
        "text-embedding-v3": 1024,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        dim_override: int | None = None,
    ):
        cfg = get_config().get("embedding", {})
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model_name = model_name or cfg.get("model_name", "text-embedding-3-small")
        self._base_url = base_url or cfg.get("base_url")
        self._batch_size = batch_size or cfg.get("batch_size", 32)
        self._normalize = normalize if normalize is not None else cfg.get("normalize", True)
        self._dim_override = dim_override
        self._client = None

    def load(self) -> None:
        if not self._api_key:
            raise EmbeddingError("API key not set for embedding provider")

    def is_loaded(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("Cannot embed empty text list")
        client = self._get_client()

        vectors_list: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                kwargs = {"model": self._model_name, "input": batch}
                if self._dim_override:
                    kwargs["dimensions"] = self._dim_override
                response = client.embeddings.create(**kwargs)
                batch_vectors = np.array(
                    [d.embedding for d in response.data], dtype=np.float32
                )
                if self._normalize:
                    norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-12, norms)
                    batch_vectors = batch_vectors / norms
                vectors_list.append(batch_vectors)
            except Exception as e:
                raise EmbeddingError(f"Embedding API error: {e}") from e

        return np.concatenate(vectors_list, axis=0)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise EmbeddingError("Cannot embed empty query")
        vec = self.embed_documents([query])
        return vec[0]

    def dim(self) -> int:
        if self._dim_override:
            return self._dim_override
        return self._DIMS.get(self._model_name, 1024)
