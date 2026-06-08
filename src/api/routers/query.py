"""查询端点：基于 Agent 的 RAG 问答与流式输出。"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api import dependencies as deps
from src.api.schemas.query import AgentStepOut, QueryRequest, QueryResponse
from src.api.dependencies import UnknownAgentProfileError
from src.llm.types import Message

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        agent = deps.get_agent(req.profile)
    except UnknownAgentProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history = None
    if req.history:
        history = [Message(role=m.role, content=m.content) for m in req.history]

    result = await agent.run(
        query=req.query,
        history=history,
        trace_id=req.trace_id,
    )

    steps_out = [
        AgentStepOut(
            step_index=s.step_index,
            tool_name=s.tool_call.name if s.tool_call else "",
            tool_arguments=s.tool_call.arguments if s.tool_call else None,
            tool_result=s.tool_result[:500] if s.tool_result else "",
            thinking=s.thinking,
            duration_ms=s.duration_ms,
        )
        for s in result.steps
    ]

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        iterations=result.iterations,
        steps=steps_out,
        processing_stages=result.processing_stages,
        trace_id=result.trace_id,
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    try:
        agent = deps.get_agent(req.profile)
    except UnknownAgentProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history = None
    if req.history:
        history = [Message(role=m.role, content=m.content) for m in req.history]

    async def event_stream():
        async for event in agent.run_stream(
            query=req.query,
            history=history,
            trace_id=req.trace_id,
        ):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")
