"""FastAPI middleware: request ID, timing, CORS."""

import time
import uuid

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger

logger = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed = time.monotonic() - start
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        return response


def register_middleware(app):
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
