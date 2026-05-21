"""Pydantic schemas for query endpoint."""

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    text: str
    document_id: str
    source_path: str
    chunk_index: int
    relevance_score: float


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
