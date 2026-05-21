# Architecture

## Overview

RAG Pipeline is a production-grade retrieval-augmented generation system for enterprise knowledge base retrieval, optimized for Chinese documents.

## Pipeline

```
Document Ingestion:  Loader → Cleaner → ChineseChunker → Embedding(BGE-M3) → Milvus
Query:              Query → Embedding → Milvus ANN Search(top-20) → Reranker(top-5) → LLM → Cited Answer
```

## Components

### Ingestion
- **DocumentLoader**: Dispatches by file extension to PDF/Markdown/Text readers
- **TextCleaner**: NFKC normalization, fullwidth-to-halfwidth, Chinese quote normalization
- **ChineseChunker**: jieba-based sentence boundary detection, structure-aware splitting, overlap

### Embedding
- **BgeM3Embedder**: BAAI/bge-m3 model, 1024-dim dense vectors, L2 normalized

### Vector Database
- **MilvusStore**: Dual-mode — Milvus Lite (zero-config dev) or Milvus Standalone (production)
- Schema: id, document_id, content, chunk_index, embedding(1024, IVF_FLAT/HNSW), metadata(JSON)

### Retrieval
- **DenseRetriever**: Embed query → ANN search in Milvus
- **BgeReranker**: BAAI/bge-reranker-v2-m3 cross-encoder scoring

### Generation
- **PromptBuilder**: Chinese system prompt with numbered context
- **CitationFormatter**: Parse [1], [2] markers → source metadata
- **Generator**: Orchestrates retrieve → rerank → generate → cite

### LLM Providers
- **ClaudeProvider**: Anthropic SDK with streaming
- **OpenAIProvider**: OpenAI SDK with streaming
- Pluggable via config: `LLM_PROVIDER=claude|openai`

### API
- **FastAPI** with async endpoints
- `POST /documents/ingest` — Upload and index
- `DELETE /documents/{id}` — Remove document
- `POST /query` — RAG Q&A
- `POST /query/stream` — SSE streaming response
- `GET /health`, `GET /health/ready` — Health checks

## Key Design Decisions

1. **Milvus Lite → Standalone upgrade**: Same SDK API, one config change
2. **Two-stage retrieval**: Dense recall (top-20) + cross-encoder rerank (top-5)
3. **Chinese-aware chunking**: jieba + sentence boundaries, no split sentences
4. **Async throughout**: Non-blocking I/O from retrieval to LLM
