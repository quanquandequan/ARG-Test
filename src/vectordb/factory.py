"""VectorDB 工厂：按配置驱动实例化。"""

from src.core.config import get_config
from src.vectordb.base import BaseVectorDB
from src.vectordb.milvus_store import MilvusStore


def get_vectordb() -> BaseVectorDB:
    cfg = get_config().get("vectordb", {})
    provider = cfg.get("provider", "milvus")

    if provider == "milvus":
        return MilvusStore()

    raise ValueError(f"Unknown vector DB provider: {provider}")
