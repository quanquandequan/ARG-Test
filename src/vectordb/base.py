"""向量数据库抽象基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    id: str
    document_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class BaseVectorDB(ABC):
    @abstractmethod
    def create_collection(self, name: str, dim: int, drop_existing: bool = False) -> None:
        ...

    @abstractmethod
    def insert(self, chunks_with_vectors: list[tuple]) -> None:
        """插入 chunks 及其 embeddings。
        每项格式：(chunk_id, document_id, content, chunk_index, embedding, metadata)
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: "np.ndarray",
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> int:
        """删除属于指定文档的所有 chunks，并返回删除数量。"""
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def drop_collection(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
