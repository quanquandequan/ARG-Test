# API Reference

Base URL: `http://localhost:8000`

## Endpoints

### `POST /documents/ingest`
Upload and index a document.

**Request**: `multipart/form-data`
- `file`: Document file (PDF, Markdown, TXT)

**Response**: `200 OK`
```json
{
  "document_id": "uuid",
  "chunk_count": 42,
  "source_path": "report.pdf"
}
```

### `DELETE /documents/{document_id}`
Delete a document and all its chunks.

**Response**: `200 OK`
```json
{
  "document_id": "uuid",
  "deleted_chunks": 42
}
```

### `POST /query`
RAG question answering.

**Request**:
```json
{
  "query": "什么是向量检索？",
  "top_k": 5,
  "filters": {"document_id": "optional-filter"},
  "stream": false
}
```

**Response**:
```json
{
  "answer": "向量检索是一种... [1]",
  "citations": [
    {
      "text": "向量检索基于...",
      "document_id": "uuid",
      "source_path": "vector-guide.md",
      "chunk_index": 3,
      "relevance_score": 0.92
    }
  ]
}
```

### `POST /query/stream`
Stream answer tokens via SSE (Server-Sent Events).

**Response**: `text/event-stream`
```
data: 向量
data: 检索
data: 是一
data: 种
data: [DONE]
```

### `GET /health`
```json
{"status": "ok", "version": "0.1.0"}
```

### `GET /health/ready`
```json
{
  "ready": true,
  "checks": {
    "embedder": true,
    "vectordb": true,
    "reranker": true
  }
}
```
