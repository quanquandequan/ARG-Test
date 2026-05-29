"""API-layer fixtures: build a FastAPI app with all fakes injected."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.dependencies as deps
from src.api.routers import health, ingestion, query


@pytest.fixture(autouse=True)
def _clear_dep_caches():
    """Ensure dependency caches are clear before and after each test."""
    deps.clear_all_caches()
    yield
    deps.clear_all_caches()


@pytest.fixture
def wired_singletons(fake_embedder, fake_vectordb, fake_llm, fake_reranker, monkeypatch):
    """Patch internal factory functions so lru_cache deps build with fakes.

    Because the routers use ``from src.api.dependencies import get_agent``
    (a local binding), we cannot monkeypatch ``deps.get_agent`` directly.
    Instead we patch the leaf factory functions that the lru_cache chain
    calls internally, then clear the caches so they rebuild with fakes.
    """
    monkeypatch.setattr(deps, "get_singleton_embedder", lambda: fake_embedder)
    monkeypatch.setattr(deps, "get_singleton_vectordb", lambda: fake_vectordb)
    monkeypatch.setattr(deps, "get_singleton_llm", lambda: fake_llm)
    monkeypatch.setattr(deps, "get_singleton_reranker", lambda: fake_reranker)

    # Clear caches so the lru_cache functions rebuild using the patched factories.
    deps.clear_all_caches()

    return {
        "embedder": fake_embedder,
        "vectordb": fake_vectordb,
        "llm": fake_llm,
        "reranker": fake_reranker,
    }


@pytest.fixture
def client(wired_singletons) -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)
    return TestClient(app)
