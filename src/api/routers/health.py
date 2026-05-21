"""Health check endpoints."""

from fastapi import APIRouter

from src.api.dependencies import _singleton_embedder, _singleton_reranker, _singleton_vectordb
from src.api.schemas.query import HealthResponse, ReadyResponse
from src.core.config import get_config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    cfg = get_config()
    return HealthResponse(status="ok", version=cfg.get("app", {}).get("version", "0.1.0"))


@router.get("/health/ready", response_model=ReadyResponse)
async def ready():
    checks: dict[str, bool] = {}

    try:
        embedder = _singleton_embedder()
        checks["embedder"] = embedder.is_loaded()
    except Exception:
        checks["embedder"] = False

    try:
        vectordb = _singleton_vectordb()
        vectordb.count()
        checks["vectordb"] = True
    except Exception:
        checks["vectordb"] = False

    try:
        reranker = _singleton_reranker()
        checks["reranker"] = reranker.is_loaded()
    except Exception:
        checks["reranker"] = False

    return ReadyResponse(ready=all(checks.values()), checks=checks)
