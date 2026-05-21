"""Embedder factory — config-driven instantiation."""

import os

from src.core.config import get_config
from src.embedding.base import BaseEmbedder
from src.embedding.bge_m3 import BgeM3Embedder
from src.embedding.openai_embedder import OpenAIEmbedder

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_embedder() -> BaseEmbedder:
    cfg = get_config().get("embedding", {})
    provider = os.environ.get("EMBEDDING_PROVIDER") or cfg.get("provider", "bge_m3")

    if provider == "bge_m3":
        return BgeM3Embedder()
    elif provider == "openai":
        return OpenAIEmbedder()
    elif provider == "dashscope":
        return OpenAIEmbedder(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            model_name=cfg.get("model_name", "text-embedding-v4"),
            base_url=_DASHSCOPE_BASE_URL,
            dim_override=cfg.get("dim"),
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
