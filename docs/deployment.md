# Deployment

## Quick Start (Development)

```bash
cd rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # edit with your API keys

# Ingest documents
python scripts/ingest_docs.py --dir ./data/documents

# Start interactive chat
rag chat

# Debug mode (shows tool calls and full logs)
rag chat -d
```

Note:
- Development defaults to `Milvus Lite` via `./data/milvus_lite/milvus.db`.
- The local KB needs the `milvus_lite` runtime, which is installed by `pymilvus[milvus_lite]`.
- If you see a Milvus Lite connection error, rerun:

```bash
pip install -e ".[dev]"
```

## Docker Compose (Milvus Standalone)

```bash
# Start Milvus stack (etcd + MinIO + Milvus)
docker compose up -d

# Monitor
docker compose logs -f milvus
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `RAG_ENV` | `development` | Config profile (development/production) |
| `LLM_PROVIDER` | `claude` | LLM backend (claude/openai) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `MILVUS_MODE` | `lite` | Milvus mode (lite/standalone) |
| `MILVUS_URI` | `./data/milvus_lite` | Milvus connection URI |

## Requirements

- Python 3.11+
- 8GB+ RAM recommended (models: ~4GB)
- Docker (for production deployment)
- Anthropic or OpenAI API key
