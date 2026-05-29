"""Shared test fixtures and config patching."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

import src.core.config as cfg_mod
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker, FakeVectorDB


def _build_test_config() -> DictConfig:
    return OmegaConf.create(
        {
            "app": {
                "name": "test",
                "version": "0.0.0-test",
                "debug": True,
                "log_level": "DEBUG",
            },
            "embedding": {
                "provider": "fake",
                "model_name": "fake",
                "device": "cpu",
                "normalize": True,
                "batch_size": 8,
                "use_onnx": False,
                "dim": 1024,
            },
            "chunking": {
                "strategy": "chinese_aware",
                "chunk_size": 64,
                "chunk_overlap": 10,
                "min_chunk_size": 8,
            },
            "vectordb": {
                "provider": "fake",
                "mode": "lite",
                "uri": "./.pytest_milvus_unused",
                "collection_name": "test_kb",
                "dim": 1024,
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "index_params": {"nlist": 32},
            },
            "retrieval": {
                "top_k": 5,
                "final_k": 3,
                "similarity_threshold": 0.0,
            },
            "reranker": {
                "provider": "fake",
                "model_name": "fake",
                "device": "cpu",
                "batch_size": 8,
            },
            "llm": {
                "provider": "fake",
                "model": "fake",
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": False,
            },
            "agent": {
                "max_iterations": 5,
                "max_history_tokens": 2000,
                "system_prompt": "test",
                "tools": ["knowledge_search"],
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "cors_origins": ["*"],
                "request_timeout": 30,
            },
            "logging": {"json_format": False, "level": "DEBUG"},
        }
    )


@pytest.fixture
def test_config() -> DictConfig:
    return _build_test_config()


@pytest.fixture(autouse=True)
def _patch_config(test_config, monkeypatch):
    """Make `get_config()` return the test config without touching YAML files."""
    monkeypatch.setattr(cfg_mod, "_CONFIG", test_config, raising=False)
    yield
    monkeypatch.setattr(cfg_mod, "_CONFIG", None, raising=False)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    emb = FakeEmbedder()
    emb.load()
    return emb


@pytest.fixture
def fake_vectordb() -> FakeVectorDB:
    return FakeVectorDB()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
