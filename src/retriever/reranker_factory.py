"""Reranker 工厂：按配置驱动实例化。"""

import os

from src.core.config import get_config
from src.retriever.dashscope_reranker import DashScopeReranker
from src.retriever.openai_reranker import OpenAIReranker
from src.retriever.reranker import BgeReranker
from src.retriever.reranker_base import BaseReranker


def get_reranker() -> BaseReranker:
    cfg = get_config().get("reranker", {})
    provider = os.environ.get("RERANKER_PROVIDER") or cfg.get("provider", "bge_reranker")

    if provider == "bge_reranker":
        return BgeReranker()
    elif provider == "openai":
        return OpenAIReranker()
    elif provider == "dashscope":
        return DashScopeReranker()

    raise ValueError(f"Unknown reranker provider: {provider}")
