"""通过 FlagEmbedding / sentence-transformers 使用 BGE-M3 embedding 模型。"""

import numpy as np

from src.core.config import get_config
from src.core.exceptions import EmbeddingError
from src.embedding.base import BaseEmbedder


class BgeM3Embedder(BaseEmbedder):
    """BAAI/bge-m3 embedding 模型，中文效果优秀，1024 维，多语言。

    支持 dense、sparse 和 ColBERT 表示。主检索链路使用 dense 向量（1024 维）。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,

    ):
        cfg = get_config().get("embedding", {})
        self._model_name = model_name or cfg.get("model_name", "BAAI/bge-m3")
        self._device = device or cfg.get("device", "cpu")
        self._batch_size = batch_size or cfg.get("batch_size", 32)
        self._normalize = normalize if normalize is not None else cfg.get("normalize", True)
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=False,
                device=self._device,
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load BGE-M3 model '{self._model_name}': {e}"
            ) from e

    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("Cannot embed empty text list")
        if self._model is None:
            self.load()

        vectors_list: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            # BGE-M3 返回包含 'dense_vecs'、'lexical_weights'、'colbert_vecs' 的字典
            output = self._model.encode(batch, batch_size=self._batch_size)
            dense = output["dense_vecs"]
            if self._normalize:
                norms = np.linalg.norm(dense, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-12, norms)
                dense = dense / norms
            vectors_list.append(dense.astype(np.float32))

        return np.concatenate(vectors_list, axis=0)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise EmbeddingError("Cannot embed empty query")
        vec = self.embed_documents([query])
        return vec[0]

    def dim(self) -> int:
        return 1024
