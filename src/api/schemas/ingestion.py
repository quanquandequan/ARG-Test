"""Pydantic schemas for document ingestion."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    source_path: str
