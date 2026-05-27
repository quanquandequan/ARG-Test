"""API-layer fixtures: build a FastAPI app with all fakes injected."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.dependencies as deps
from src.agent.react_loop import ReActAgent
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.api.routers import health, ingestion, query
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine


@pytest.fixture
def wired_singletons(fake_embedder, fake_vectordb, fake_llm, fake_reranker, monkeypatch):
    """Force `dependencies` module-level singletons to the fakes."""
    monkeypatch.setattr(deps, "_embedder", fake_embedder, raising=False)
    monkeypatch.setattr(deps, "_vectordb", fake_vectordb, raising=False)
    monkeypatch.setattr(deps, "_llm", fake_llm, raising=False)
    monkeypatch.setattr(deps, "_reranker", fake_reranker, raising=False)

    dense = DenseRetriever(fake_embedder, fake_vectordb)
    engine = RetrievalEngine(dense_retriever=dense, reranker=fake_reranker)
    monkeypatch.setattr(deps, "_retrieval_engine", engine, raising=False)

    tools = [
        KnowledgeBaseTool(engine),
        WebSearchTool(),
    ]
    agent = ReActAgent(llm=fake_llm, tools=tools, system_prompt="test")
    monkeypatch.setattr(deps, "_agent", agent, raising=False)

    return {
        "embedder": fake_embedder,
        "vectordb": fake_vectordb,
        "llm": fake_llm,
        "reranker": fake_reranker,
        "retrieval_engine": engine,
        "agent": agent,
    }


@pytest.fixture
def client(wired_singletons) -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)
    return TestClient(app)
