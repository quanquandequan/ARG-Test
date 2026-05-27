# Architecture

## Overview

RAG Pipeline is a production-grade **Agentic** retrieval-augmented generation system for enterprise knowledge base Q&A, optimized for Chinese documents.

The system is built around a **ReAct Agent** that drives the full reasoning loop. RAG retrieval is exposed as a **tool** the Agent can call — not a hardwired pipeline.

## Agent Architecture

```
User Query
    │
    ▼
ReActAgent  ── Think → Act → Observe (up to max_iterations) ──▶  Final Answer
    │
    ├── KnowledgeBaseTool ──▶ Generator (embed → ANN search → rerank) ──▶ Milvus
    │
    └── WebSearchTool ──▶ External search (best-effort fallback)
    │
    ▼
LLM Provider  (Claude / OpenAI / DeepSeek — unified tool-calling interface)
```

## Document Ingestion (unchanged)

```
Loader → Cleaner → ChineseChunker → Embedding → Milvus
```

## Components

### Agent Layer (`src/agent/`)

| Class | Role |
|-------|------|
| `ReActAgent` | ReAct Think→Act→Observe loop; concurrent tool execution via `asyncio.gather` |
| `ToolRegistry` | Register / lookup tools; emit JSON Schema for LLM function-calling |
| `KnowledgeBaseTool` | Wraps `Generator.search()` — retrieves and formats ranked chunks |
| `WebSearchTool` | Best-effort DuckDuckGo HTML scraping (external fallback) |
| `BaseTool` | Abstract interface — `name`, `description`, `parameters` (JSON Schema), `execute()` |

Agent configuration lives in **`configs/default.yaml`** under `agent:`:

```yaml
agent:
  max_iterations: 10
  system_prompt: |
    ...
```

### Retrieval Engine (`src/generation/generator.py`)

`Generator.search()` = embed query → ANN search (top-20) → cross-encoder rerank (top-5) → `list[SearchResult]`

It does **not** call the LLM; generation is entirely the Agent's responsibility.

### LLM Providers (`src/llm/`)

- **ClaudeProvider** — Anthropic SDK, full tool-calling support
- **OpenAIProvider** — OpenAI-compatible (OpenAI, DeepSeek, DashScope), tool-calling support

Both implement `BaseLLM.generate_chat(messages, tools, tool_choice)` → `ChatResponse`.
Internal message format (`Message`, `ToolCall`, `ChatResponse`, `ContentBlock`) is provider-agnostic; each provider handles serialisation to its own API format.

### Ingestion (`src/ingestion/`)

- **DocumentLoader** — dispatches by extension (PDF / Markdown / Text / XLSX / XMind)
- **TextCleaner** — NFKC normalization, fullwidth→halfwidth, Chinese quote normalization
- **ChineseChunker** — jieba sentence-boundary detection, overlap, no mid-sentence splits

### Embedding & Vector DB

- **OpenAIEmbedder** / **BgeM3Embedder** — pluggable via `embedding.provider` config
- **MilvusStore** — Lite (zero-config dev) or Standalone (production); same SDK API

### API (`src/api/`)

| Endpoint | Description |
|----------|-------------|
| `POST /query` | Agent Q&A — returns `answer`, `citations`, `iterations`, `steps` |
| `POST /query/stream` | SSE streaming: `tool_call` / `tool_result` / `answer` events |
| `POST /documents/ingest` | Upload and index a document |
| `DELETE /documents/{id}` | Remove all chunks for a document |
| `GET /health` | Liveness probe |
| `GET /health/ready` | Readiness probe (embedder + vectordb + reranker + llm) |

### CLI (`src/agent/cli.py`)

```bash
rag ask "问题"           # single query
rag ask -s "问题"        # streaming output
rag ask -v "问题"        # verbose: show tool calls and step trace
rag chat                 # interactive multi-turn chat
```

## Key Design Decisions

1. **Agent-first**: LLM drives the reasoning loop; RAG is a callable tool, not the pipeline
2. **Concurrent tool execution**: multiple tool calls in one LLM response run via `asyncio.gather`
3. **Two-stage retrieval**: dense recall (top-20) + cross-encoder rerank (top-5)
4. **Unified tool protocol**: `BaseTool.to_openai_schema()` + per-provider conversion in LLM layer
5. **Multi-provider LLM**: same Agent code works with Claude, OpenAI, DeepSeek
6. **Chinese-aware chunking**: jieba + sentence boundaries — no mid-sentence splits
7. **Config-driven Agent**: system prompt, max iterations, and tool list are YAML-configurable
8. **Milvus Lite → Standalone**: same SDK, one config change
