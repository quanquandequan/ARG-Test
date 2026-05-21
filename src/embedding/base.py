"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents. Returns (N, dim) float32 array."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) float32 array."""
        ...

    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (lazy initialization)."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...
