"""Shared test fixtures."""

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def test_config():
    return OmegaConf.create({
        "app": {"name": "test", "debug": True, "log_level": "DEBUG"},
        "embedding": {
            "provider": "bge_m3",
            "model_name": "BAAI/bge-m3",
            "device": "cpu",
            "normalize": True,
            "batch_size": 8,
            "use_onnx": False,
        },
        "chunking": {
            "strategy": "chinese_aware",
            "chunk_size": 512,
            "chunk_overlap": 100,
            "min_chunk_size": 50,
        },
        "vectordb": {
            "provider": "milvus",
            "mode": "lite",
            "uri": "./data/milvus_lite_test",
            "collection_name": "test_kb",
            "dim": 1024,
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
        },
        "retrieval": {"top_k": 20, "final_k": 5, "similarity_threshold": 0.3},
        "reranker": {
            "provider": "bge_reranker",
            "model_name": "BAAI/bge-reranker-v2-m3",
            "device": "cpu",
        },
        "llm": {
            "provider": "claude",
            "model": "claude-sonnet-4-6",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    })
