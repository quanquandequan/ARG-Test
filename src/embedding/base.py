"""Embedding 模型抽象基类。"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """对一批文档做 embedding，返回 (N, dim) float32 数组。"""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """对单个 query 做 embedding，返回 (dim,) float32 数组。"""
        ...

    @abstractmethod
    def dim(self) -> int:
        """返回 embedding 维度。"""
        ...

    @abstractmethod
    def load(self) -> None:
        """将模型加载到内存（延迟初始化）。"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        ...
