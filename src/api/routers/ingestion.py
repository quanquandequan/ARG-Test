"""Document ingestion endpoints."""

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.api.dependencies import _singleton_embedder, _singleton_vectordb
from src.api.schemas.ingestion import IngestResponse
from src.core.exceptions import IngestionError
from src.core.logging import get_logger
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["ingestion"])
logger = get_logger(__name__)


def _get_pipeline():
    return IngestionPipeline(
        loader=DocumentLoader(),
        cleaner=TextCleaner(),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        pipeline = _get_pipeline()
        doc, chunks = pipeline.ingest(tmp_path)

        embedder = _singleton_embedder()
        chunk_texts = [c.content for c in chunks]
        vectors = embedder.embed_documents(chunk_texts)

        vectordb = _singleton_vectordb()
        rows = []
        for chunk, vec in zip(chunks, vectors):
            rows.append((
                chunk.id,
                chunk.document_id,
                chunk.content,
                chunk.chunk_index,
                vec,
                {"source_path": str(tmp_path), "chunk_index": chunk.chunk_index},
            ))
        vectordb.insert(rows)

        logger.info("document_ingested", doc_id=doc.id, chunks=len(chunks))
        return IngestResponse(
            document_id=doc.id,
            chunk_count=len(chunks),
            source_path=file.filename or str(tmp_path),
        )

    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    vectordb = _singleton_vectordb()
    deleted = vectordb.delete_by_document_id(document_id)
    return {"document_id": document_id, "deleted_chunks": deleted}
