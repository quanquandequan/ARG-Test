"""Abstract base class for vector database."""

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
        """Insert chunks with their embeddings.
        Each item: (chunk_id, document_id, content, chunk_index, embedding, metadata)
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
        """Delete all chunks belonging to a document. Returns count deleted."""
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
