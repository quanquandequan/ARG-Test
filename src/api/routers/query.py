"""Query endpoints — RAG Q&A and streaming."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_generator
from src.api.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    gen = get_generator()
    result = await gen.query(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
    )
    return QueryResponse(
        answer=result.answer,
        citations=[
            {
                "text": c.text,
                "document_id": c.document_id,
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "relevance_score": c.relevance_score,
            }
            for c in result.citations
        ],
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    gen = get_generator()

    async def event_stream():
        async for token in gen.query_stream(
            query=req.query,
            top_k=req.top_k,
            filters=req.filters,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
