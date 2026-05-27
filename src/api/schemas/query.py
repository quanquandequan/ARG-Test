"""Pydantic schemas for query endpoint."""

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    text: str = ""
    document_id: str = ""
    source_path: str = ""
    chunk_index: int = 0
    relevance_score: float = 0.0
    index: int | None = None


class AgentStepOut(BaseModel):
    step_index: int
    tool_name: str = ""
    tool_arguments: dict | None = None
    tool_result: str = ""
    thinking: str = ""


class MessageSchema(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None
    stream: bool = False
    max_iterations: int = Field(default=10, ge=1, le=30)
    history: list[MessageSchema] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    iterations: int = 0
    steps: list[AgentStepOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
