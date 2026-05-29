"""Document ingestion endpoints."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api import dependencies as deps
from src.api.schemas.ingestion import IngestResponse
from src.core.exceptions import IngestionError, RAGPipelineError
from src.core.logging import get_logger
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["ingestion"])
logger = get_logger(__name__)


def _get_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        loader=DocumentLoader(),
        cleaner=TextCleaner(),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    inserted_doc_id: str | None = None
    vectordb = None
    try:
        pipeline = _get_pipeline()
        doc, chunks = pipeline.ingest(tmp_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No content chunks produced")

        embedder = deps.get_singleton_embedder()
        chunk_texts = [c.content for c in chunks]
        vectors = embedder.embed_documents(chunk_texts)

        vectordb = deps.get_singleton_vectordb()
        rows = []
        for chunk, vec in zip(chunks, vectors):
            rows.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    vec,
                    {
                        "source_path": file.filename or str(tmp_path),
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )
        vectordb.insert(rows)
        inserted_doc_id = doc.id

        logger.info("document_ingested", doc_id=doc.id, chunks=len(chunks))
        return IngestResponse(
            document_id=doc.id,
            chunk_count=len(chunks),
            source_path=file.filename or str(tmp_path),
        )

    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except RAGPipelineError as e:
        _rollback(vectordb, inserted_doc_id)
        logger.exception("ingest_pipeline_error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        _rollback(vectordb, inserted_doc_id)
        logger.exception("ingest_unexpected_error")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


def _rollback(vectordb, doc_id: str | None) -> None:
    if vectordb is None or not doc_id:
        return
    try:
        deleted = vectordb.delete_by_document_id(doc_id)
        if deleted:
            logger.warning("ingest_rolled_back", doc_id=doc_id, deleted_chunks=deleted)
    except Exception:
        logger.exception("ingest_rollback_failed", doc_id=doc_id)


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    vectordb = deps.get_singleton_vectordb()
    deleted = vectordb.delete_by_document_id(document_id)
    return {"document_id": document_id, "deleted_chunks": deleted}
