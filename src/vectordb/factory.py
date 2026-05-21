"""VectorDB factory — config-driven instantiation."""

from src.core.config import get_config
from src.vectordb.base import BaseVectorDB
from src.vectordb.milvus_store import MilvusStore


def get_vectordb() -> BaseVectorDB:
    cfg = get_config().get("vectordb", {})
    provider = cfg.get("provider", "milvus")

    if provider == "milvus":
        return MilvusStore()

    raise ValueError(f"Unknown vector DB provider: {provider}")
