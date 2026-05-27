"""Query endpoints — Agent-based RAG Q&A and streaming."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_agent
from src.api.schemas.query import AgentStepOut, QueryRequest, QueryResponse
from src.llm.types import Message

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    agent = get_agent()

    history = None
    if req.history:
        history = [Message(role=m.role, content=m.content) for m in req.history]

    result = await agent.run(
        query=req.query,
        history=history,
    )

    steps_out = [
        AgentStepOut(
            step_index=s.step_index,
            tool_name=s.tool_call.name if s.tool_call else "",
            tool_arguments=s.tool_call.arguments if s.tool_call else None,
            tool_result=s.tool_result[:500] if s.tool_result else "",
            thinking=s.thinking,
        )
        for s in result.steps
    ]

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        iterations=result.iterations,
        steps=steps_out,
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    agent = get_agent()

    history = None
    if req.history:
        history = [Message(role=m.role, content=m.content) for m in req.history]

    async def event_stream():
        async for event in agent.run_stream(query=req.query, history=history):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")
