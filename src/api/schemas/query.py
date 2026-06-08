"""查询端点的 Pydantic schema。"""

from pydantic import BaseModel, Field

from src.agent.types import Citation


class AgentStepOut(BaseModel):
    step_index: int
    tool_name: str = ""
    tool_arguments: dict | None = None
    tool_result: str = ""
    thinking: str = ""
    duration_ms: float = 0.0


class MessageSchema(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    profile: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None
    stream: bool = False
    max_iterations: int = Field(default=10, ge=1, le=30)
    history: list[MessageSchema] | None = None
    trace_id: str | None = None  # 允许客户端传入自己的 trace-id


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    iterations: int = 0
    steps: list[AgentStepOut] = Field(default_factory=list)
    processing_stages: dict[str, float] = Field(default_factory=dict)
    trace_id: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
