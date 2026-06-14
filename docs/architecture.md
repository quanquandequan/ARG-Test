# Architecture

## Overview

**RAG Pipeline** is a production-grade **Agentic** RAG system for enterprise knowledge base Q&A, optimised for Chinese documents.

The architecture has two main layers:

1. **Ingestion** — documents are cleaned, chunked, embedded, and stored in Milvus.
2. **Agent** — a ReAct loop that drives multi-step reasoning; RAG retrieval is one callable *tool*.

---

## High-Level Diagram

```
┌─────────────────────────── User ───────────────────────────────┐
│  CLI: rag chat                CLI: rag chat -d (调试模式)       │
└───────────────────────────────┬────────────────────────────────┘
                                │  query + history
                                ▼
┌──────────────────────── ReActAgent ────────────────────────────┐
│                                                                │
│  ┌─ Think ─────────────────────────────────────────────────┐  │
│  │  LLM.generate_chat(messages, tools, tool_choice)        │  │
│  │  ← ChatResponse { content, stop_reason, tool_calls }    │  │
│  └─────────────────────────────────────────────────────────┘  │
│           │ stop_reason == "tool_use"?                         │
│      Yes  ▼                                    No ▼            │
│  ┌─ Act ───────────────────┐          ┌─ Answer ────────────┐  │
│  │  asyncio.gather(        │          │  extract_citations  │  │
│  │    tool1.execute(),     │          │  return AgentResult │  │
│  │    tool2.execute(), …   │          └────────────────────┘  │
│  │  ) — concurrent         │                                  │
│  └─────────┬───────────────┘                                  │
│            │                                                  │
│  ┌─ Observe ────────────────────────────────────────────────┐  │
│  │  Append tool results to messages; repeat up to N iters   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  History sliding-window truncation (token budget):            │
│    keep_last=4 unconditional + fill backwards to max_tokens   │
│                                                                │
│  Observability:                                               │
│    • trace_id  (UUID, client-passable, attached to all logs)  │
│    • per-step duration_ms  (LLM + tool wall-clock)            │
│    • processing_stages  { "iter0.search_knowledge": 82.4,     │
│                            "total": 1204.1 }                  │
└────────────────────────────────────────────────────────────────┘
         │                         │
         ▼                         ▼
┌── SearchKnowledgeTool ─────────┐  ┌── Other Facades ─────────────────────┐
│  KB first, Web on demand       │  │  AnalyzeRequirementTool              │
│  wraps KnowledgeBaseTool + Web │  │  DesignTestCasesTool                 │
│                                 │  │  ExecuteScenarioTool                │
└──────────┬──────────────────────┘  └─────────────────────────────────────┘
           │
           ▼
    RetrievalEngine → Milvus vector store
    (Lite for dev / Standalone for prod)
```

---

## Agent Tool Protocol

Every tool implements `BaseTool`:

```python
class BaseTool(ABC):
    @property
    def name(self) -> str: ...         # must be unique

    @property
    def description(self) -> str: ... # shown to LLM

    @property
    def parameters(self) -> dict: ... # JSON Schema (OpenAI function-call format)

    async def execute(self, **kwargs) -> str: ...

    def to_tool_schema(self) -> dict:
        return {"name": ..., "description": ..., "parameters": ...}
```

`ToolRegistry.definitions()` returns the schema list; each LLM provider converts it to its own wire format (`_tools_to_anthropic` / `_tools_to_openai`).

工具列表由 `configs/default.yaml` 的 `agent.tools` 统一管理：

```yaml
agent:
  tools:
    - search_knowledge
    - analyze_requirement
    - design_test_cases
    - execute_scenario
    - device_tool
    - screen_tool
    - action_tool
    - assertion_tool
```

---

## ReAct Loop Detail

```
Iteration i:
  1. generate_chat(messages, tool_defs) → response
  2. if response.stop_reason == "tool_use":
       asyncio.gather(tool1.execute(), tool2.execute(), …)  ← concurrent
       append tool results → messages
       continue to iteration i+1
  3. else:
       extract [N] citations from response.content
       return AgentResult(answer, steps, citations, processing_stages, trace_id)

If max_iterations reached:
  force one final generate_chat without tools → summarise
```

---

## Retrieval Engine

`RetrievalEngine.search(query, top_k, final_k, filters)`:

```
query ──► Embedder.embed_query() ──► DenseRetriever.retrieve(top_k=20)
                                          │
                                          ▼
                                   Milvus cosine ANN
                                          │
                                          ▼
                                   BaseReranker.rerank(top_k=5)
                                          │
                                          ▼
                                   list[SearchResult]
```

LLM generation is **not** done here; that is the Agent's responsibility.

---

## LLM Providers

| Provider | Class | Features |
|----------|-------|---------|
| Anthropic Claude | `ClaudeProvider` | `generate_chat`, `generate_chat_stream`, tool-calling |
| OpenAI / DeepSeek / DashScope | `OpenAIProvider` | same interface, `base_url` configurable |

Message conversion is handled inside each provider (`_messages_to_anthropic`, `_messages_to_openai`, etc.).

---

## History Truncation

`src/agent/history.py::truncate_history(history, max_tokens, keep_last)`:

1. Always keep the last `keep_last=4` messages unconditionally.
2. Fill backwards from the oldest of the remaining slice while the cumulative token estimate (`len(content) / 2.5`) stays below `max_tokens` (default 4 000).
3. System prompt and the current user query are added by the caller — they are **not** in `history`.

---

## Document Ingestion

```
DocumentLoader   (PDF / MD / TXT / XLSX / XMind)
      │
      ▼
TextCleaner      (NFKC, fullwidth→halfwidth, Chinese quote normalisation)
      │
      ▼
ChineseChunker   (jieba sentence boundary, chunk_size, overlap)
      │
      ▼
Embedder.embed_documents()
      │
      ▼
MilvusStore.insert()
```

---

## CLI

```bash
rag chat                        # 交互式对话（精简模式，无日志）
rag chat -d                     # 调试模式（显示工具调用与详细日志）
rag --env production chat       # 使用 production 配置
```

---

## Configuration

All runtime behaviour is controlled by `configs/default.yaml` (override per environment):

```yaml
agent:
  max_iterations: 10
  max_history_tokens: 4000    # sliding-window token budget
  system_prompt_id: agent     # 指向 prompts.agent.system_prompt
  tools:
    - search_knowledge
    - analyze_requirement
    - design_test_cases
    - execute_scenario
    - device_tool
    - screen_tool
    - action_tool
    - assertion_tool

retrieval:
  top_k: 20    # ANN candidates
  final_k: 5   # after reranking

llm:
  provider: deepseek
  model: deepseek-chat
  temperature: 0.3
  max_tokens: 2048
```

---

## Key Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Agent-first | LLM drives reasoning; RAG is a tool, not the pipeline |
| 2 | Concurrent tool execution | `asyncio.gather` — multiple tool calls in one iteration run in parallel |
| 3 | Config-driven tool list | Add/remove tools in YAML without code changes |
| 4 | Sliding-window history | Prevents context overflow without a per-provider tokeniser |
| 5 | trace_id end-to-end | UUID propagated through logs for distributed tracing |
| 6 | per-step `duration_ms` | surfaced in `AgentStep` and `processing_stages` for latency observability |
| 7 | WebSearch best-effort | Documented as scraping — replace with stable API for production |
| 8 | Two-stage retrieval | Dense recall (top-20) + cross-encoder rerank (top-5) |
