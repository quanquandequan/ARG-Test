"""健康检查端点。"""

from fastapi import APIRouter

from src.api import dependencies as deps
from src.api.schemas.query import HealthResponse, ReadyResponse
from src.core.config import get_config
from src.core.logging import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health():
    cfg = get_config()
    return HealthResponse(status="ok", version=cfg.get("app", {}).get("version", "0.1.0"))


@router.get("/health/ready", response_model=ReadyResponse)
async def ready():
    checks: dict[str, bool] = {}

    try:
        embedder = deps.get_singleton_embedder()
        checks["embedder"] = embedder.is_loaded()
    except Exception:
        logger.exception("ready_embedder_failed")
        checks["embedder"] = False

    try:
        vectordb = deps.get_singleton_vectordb()
        vectordb.count()
        checks["vectordb"] = True
    except Exception:
        logger.exception("ready_vectordb_failed")
        checks["vectordb"] = False

    try:
        reranker = deps.get_singleton_reranker()
        # 对基于 API 的 reranker（OpenAI、DashScope），is_loaded() 只检查
        # API key 环境变量是否已设置，并不验证远程连通性。
        # 对本地 BgeReranker，则检查模型权重是否在内存中。
        checks["reranker"] = reranker.is_loaded()
    except Exception:
        logger.exception("ready_reranker_failed")
        checks["reranker"] = False

    try:
        llm = deps.get_singleton_llm()
        # 只检查 provider 已构造，不发起远程调用。
        checks["llm"] = llm is not None
    except Exception:
        logger.exception("ready_llm_failed")
        checks["llm"] = False

    return ReadyResponse(ready=all(checks.values()), checks=checks)
