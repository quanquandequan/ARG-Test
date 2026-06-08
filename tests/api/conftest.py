"""API 层 fixture：构建注入所有 fake 的 FastAPI app。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.dependencies as deps
from src.bootstrap import AppContainer
from src.api.routers import health, ingestion, query


@pytest.fixture(autouse=True)
def _clear_dep_caches():
    """确保每个测试前后都清理依赖缓存。"""
    deps.clear_all_caches()
    yield
    deps.clear_all_caches()


@pytest.fixture
def wired_singletons(fake_embedder, fake_vectordb, fake_llm, fake_reranker, monkeypatch):
    """补丁内部工厂函数，使 lru_cache 依赖使用 fake 构建。

    由于 routers 使用 ``from src.api.dependencies import get_agent``
    （本地绑定），不能直接 monkeypatch ``deps.get_agent``。
    因此改为补丁 lru_cache 链内部调用的叶子工厂函数，
    再清理缓存，让它们用 fake 重新构建。
    """
    container = AppContainer(
        _embedder=fake_embedder,
        _vectordb=fake_vectordb,
        _llm=fake_llm,
        _reranker=fake_reranker,
    )
    monkeypatch.setattr(deps, "get_container", lambda: container)

    # 清理缓存，让 lru_cache 函数使用打过补丁的工厂重新构建。
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
