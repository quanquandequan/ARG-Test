"""Milvus 向量存储，支持 Lite（开发）和 Standalone（生产）模式。"""

import threading

import numpy as np

from src.core.config import get_config
from src.core.exceptions import CollectionNotFoundError, ConnectionError
from src.core.logging import get_logger
from src.vectordb.base import BaseVectorDB, SearchResult

logger = get_logger(__name__)


class MilvusStore(BaseVectorDB):
    def __init__(
        self,
        uri: str | None = None,
        mode: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
    ):
        cfg = get_config().get("vectordb", {})
        self._mode = mode or cfg.get("mode", "lite")
        self._uri = uri or cfg.get("uri", "./data/milvus_lite")
        self._collection_name = collection_name or cfg.get("collection_name", "knowledge_base")
        self._dim = dim or cfg.get("dim", 1024)
        self._index_type = cfg.get("index_type", "IVF_FLAT")
        self._metric_type = cfg.get("metric_type", "COSINE")
        self._index_params = dict(cfg.get("index_params", {"nlist": 128}))
        self._client = None
        # 多线程写入保护锁（MilvusLite 单文件不支持并发写）
        self._lock = threading.Lock()

    def _connect(self) -> None:
        # 快速路径：已连接则直接返回
        if self._client is not None:
            return
        # 双重检查锁：避免多线程并发重复连接 pymilvus 全局 connections 单例
        with self._lock:
            if self._client is not None:
                return
            try:
                from pymilvus import connections
                if self._mode == "lite":
                    connections.connect(alias="default", uri=self._uri)
                else:
                    host = get_config().get("vectordb", {}).get("host", "localhost")
                    port = get_config().get("vectordb", {}).get("port", 19530)
                    connections.connect(alias="default", host=host, port=port)
                self._client = "default"
                logger.info("milvus_connected", mode=self._mode, uri=self._uri)
            except ModuleNotFoundError as e:
                if self._mode == "lite" and e.name == "milvus_lite":
                    raise ConnectionError(
                        "Failed to connect to Milvus Lite: missing 'milvus_lite' runtime. "
                        "Install dependencies with: pip install -e \".[dev]\" "
                        "or pip install \"pymilvus[milvus_lite]>=2.4\""
                    ) from e
                raise ConnectionError(f"Failed to connect to Milvus: {e}") from e
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Milvus: {e}") from e

    def _ensure_collection_exists(self, name: str) -> None:
        from pymilvus import utility

        if not utility.has_collection(name):
            raise CollectionNotFoundError(
                f"Milvus collection '{name}' does not exist. "
                "Please ingest documents before querying the knowledge base."
            )

    def create_collection(self, name: str | None = None, dim: int | None = None, drop_existing: bool = False) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        self._connect()
        name = name or self._collection_name
        dim = dim or self._dim
        self._collection_name = name

        if utility.has_collection(name):
            if drop_existing:
                utility.drop_collection(name)
            else:
                logger.info("collection_exists", name=name)
                self._ensure_loaded(name)
                return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="RAG knowledge base")

        collection = Collection(name, schema)
        index_params = {
            "metric_type": self._metric_type,
            "index_type": self._index_type,
            "params": self._index_params,
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        self._ensure_loaded(name)
        logger.info("collection_created", name=name, dim=dim)

    def _ensure_loaded(self, name: str) -> None:
        from pymilvus import Collection

        coll = Collection(name)
        try:
            coll.load()
        except NotImplementedError:
            pass  # milvus-lite 会自动加载

    def insert(self, chunks_with_vectors: list[tuple]) -> None:
        from pymilvus import Collection

        self._connect()
        self._ensure_collection_exists(self._collection_name)

        ids, doc_ids, contents, indices, vectors, metadatas = [], [], [], [], [], []
        for cid, doc_id, content, chunk_idx, vec, meta in chunks_with_vectors:
            ids.append(cid)
            doc_ids.append(doc_id)
            contents.append(content)
            indices.append(chunk_idx)
            vectors.append(vec.tolist() if isinstance(vec, np.ndarray) else vec)
            metadatas.append(meta)

        # 持锁写入：MilvusLite 单文件不支持并发 insert+flush，序列化保证数据安全
        with self._lock:
            collection = Collection(self._collection_name)
            collection.insert([ids, doc_ids, contents, indices, vectors, metadatas])
            collection.flush()
        logger.debug("milvus_inserted", count=len(ids))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        from pymilvus import Collection

        self._connect()
        self._ensure_collection_exists(self._collection_name)
        collection = Collection(self._collection_name)
        self._ensure_loaded(self._collection_name)

        search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": 16},
        }
        expr = self._build_filter_expr(filters) if filters else None

        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["document_id", "content", "chunk_index", "metadata"],
        )

        return [
            SearchResult(
                id=hit.id,
                document_id=hit.entity.get("document_id", ""),
                content=hit.entity.get("content", ""),
                score=hit.distance,
                metadata=hit.entity.get("metadata", {}),
            )
            for hit in results[0]
        ]

    def delete_by_document_id(self, document_id: str) -> int:
        from pymilvus import Collection

        self._connect()
        self._ensure_collection_exists(self._collection_name)
        collection = Collection(self._collection_name)
        count_before = collection.num_entities
        collection.delete(f'document_id == "{document_id}"')
        collection.flush()
        return count_before - collection.num_entities

    def count(self) -> int:
        from pymilvus import Collection

        self._connect()
        self._ensure_collection_exists(self._collection_name)
        return Collection(self._collection_name).num_entities

    def drop_collection(self) -> None:
        from pymilvus import utility

        self._connect()
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
            logger.info("collection_dropped", name=self._collection_name)

    def close(self) -> None:
        from pymilvus import connections

        try:
            connections.disconnect("default")
        except Exception:
            pass
        self._client = None

    def _build_filter_expr(self, filters: dict) -> str:
        """根据字典构建 Milvus 标量过滤表达式。"""
        parts: list[str] = []
        for key, value in filters.items():
            if isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                parts.append(f"{key} == {value}")
        return " and ".join(parts)
