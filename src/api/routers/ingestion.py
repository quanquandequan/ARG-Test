"""文档摄取端点。"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api import dependencies as deps
from src.api.schemas.ingestion import IngestResponse
from src.core.exceptions import IngestionError, RAGPipelineError
from src.core.logging import get_logger

router = APIRouter(prefix="/documents", tags=["ingestion"])
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        service = deps.get_document_ingestion_service()
        result = await service.ingest_upload(file.filename or "upload", content)
        if result.chunk_count <= 0:
            raise HTTPException(status_code=400, detail="No content chunks produced")

        logger.info("document_ingested", doc_id=result.document_id, chunks=result.chunk_count)
        return IngestResponse(
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            source_path=result.source_path,
        )

    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except RAGPipelineError as e:
        logger.exception("ingest_pipeline_error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("ingest_unexpected_error")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    deleted = deps.get_document_ingestion_service().delete_document(document_id)
    return {"document_id": document_id, "deleted_chunks": deleted}
