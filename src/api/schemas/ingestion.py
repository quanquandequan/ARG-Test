"""Pydantic schemas for document ingestion."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    source_path: str


class DocumentInfo(BaseModel):
    id: str
    source_path: str
    chunk_count: int
    metadata: dict


class IngestError(BaseModel):
    path: str
    error: str


class BatchIngestResponse(BaseModel):
    ingested: list[IngestResponse]
    errors: list[IngestError]
