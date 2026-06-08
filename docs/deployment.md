# Deployment

## Quick Start (Development)

```bash
cd rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # edit with your API keys

# Start API with auto-reload
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Ingest documents
python scripts/ingest_docs.py --dir ./data/documents

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题"}'
```

Note:
- Development defaults to `Milvus Lite` via `./data/milvus_lite/milvus.db`.
- The local KB needs the `milvus_lite` runtime, which is installed by `pymilvus[milvus_lite]`.
- If you see a Milvus Lite connection error, rerun:

```bash
pip install -e ".[dev]"
```

## Docker Compose (Production)

```bash
# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Start stack (Milvus + etcd + MinIO + API)
docker compose up -d

# Monitor
docker compose logs -f api
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
