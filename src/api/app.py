"""FastAPI application factory."""

from fastapi import FastAPI

from src.api.middleware import register_middleware
from src.api.routers import health, ingestion, query, requirements, test_cases
from src.core.config import get_config, load_config
from src.core.logging import setup_logging


def create_app(env: str | None = None) -> FastAPI:
    load_config(env)

    cfg = get_config()
    cfg_app = cfg.get("app", {})

    app = FastAPI(
        title=cfg_app.get("name", "rag-pipeline"),
        version=cfg_app.get("version", "0.1.0"),
        description="RAG Pipeline — Enterprise Knowledge Base Retrieval",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    setup_logging()
    register_middleware(app)

    app.include_router(health.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)
    app.include_router(test_cases.router)
    app.include_router(requirements.router)

    return app
