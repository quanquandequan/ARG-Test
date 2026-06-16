# 架构说明

## 总览

**RAG Pipeline** 是一个面向企业知识库问答的生产级 **Agentic RAG** 系统，针对中文文档做了专门优化。

整体架构分两层：

1. **数据入库（Ingestion）**——文档清洗、分块、向量化，写入 Milvus。
2. **Agent**——一个驱动多步推理的 ReAct 循环；RAG 检索只是其中一个可调用的*工具*。

---

## 整体架构图

```
┌─────────────────────────── 用户 ─────────────────────────────────┐
│  CLI: rag chat                CLI: rag chat -d（调试模式）        │
└───────────────────────────────┬────────────────────────────────┘
                                │  query + history
                                ▼
┌──────────────────────── ReActAgent ────────────────────────────┐
│                                                                │
│  ┌─ Think（思考）───────────────────────────────────────────┐  │
│  │  LLM.generate_chat(messages, tools, tool_choice)        │  │
│  │  ← ChatResponse { content, stop_reason, tool_calls }    │  │
│  └─────────────────────────────────────────────────────────┘  │
│           │ stop_reason == "tool_use"？                         │
│      是   ▼                                    否 ▼            │
│  ┌─ Act（行动）────────────┐          ┌─ Answer（回答）─────┐  │
│  │  asyncio.gather(        │          │  抽取 [N] 引用标记   │  │
│  │    tool1.execute(),     │          │  返回 AgentResult   │  │
│  │    tool2.execute(), …   │          └────────────────────┘  │
│  │  ) —— 并发执行           │                                  │
│  └─────────┬───────────────┘                                  │
│            │                                                  │
│  ┌─ Observe（观察）────────────────────────────────────────┐  │
│  │  把工具结果追加进 messages；最多重复 N 轮                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  历史滑窗截断（token 预算）：                                    │
│    无条件保留最后 4 条 + 向前回填直到达到 max_tokens             │
│                                                                │
│  可观测性：                                                     │
│    • trace_id（UUID，客户端可传入，贯穿全部日志）                │
│    • 每步 duration_ms（LLM + 工具的实际耗时）                   │
│    • processing_stages { "iter0.search_knowledge": 82.4,      │
│                            "total": 1204.1 }                  │
└────────────────────────────────────────────────────────────────┘
         │                         │
         ▼                         ▼
┌── SearchKnowledgeTool ─────────┐  ┌── 其他工具 Facade ───────────────────┐
│  优先知识库，按需补充网页       │  │  AnalyzeRequirementTool              │
│  内部封装 KnowledgeBaseTool     │  │  DesignTestCasesTool                 │
│  + WebSearchTool               │  │  ExecuteScenarioTool                │
└──────────┬──────────────────────┘  └─────────────────────────────────────┘
           │
           ▼
    RetrievalEngine → Milvus 向量库
    （Lite 用于开发环境 / Standalone 用于生产环境）
```

---

## Agent 工具协议

每个工具都实现 `BaseTool`：

```python
class BaseTool(ABC):
    @property
    def name(self) -> str: ...         # 必须唯一

    @property
    def description(self) -> str: ... # 展示给 LLM 看

    @property
    def parameters(self) -> dict: ... # JSON Schema（OpenAI function-call 格式）

    async def execute(self, **kwargs) -> str: ...

    def to_tool_schema(self) -> dict:
        return {"name": ..., "description": ..., "parameters": ...}
```

`ToolRegistry.definitions()` 返回 schema 列表；每个 LLM provider 再各自转换成自己的协议格式（`_tools_to_anthropic` / `_tools_to_openai`）。

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

## ReAct 循环细节

```
第 i 轮：
  1. generate_chat(messages, tool_defs) → response
  2. 若 response.stop_reason == "tool_use"：
       asyncio.gather(tool1.execute(), tool2.execute(), …)  ← 并发执行
       把工具结果追加进 messages
       进入第 i+1 轮
  3. 否则（模型给出了最终答案，没有调用工具）：
       └─ 安全网检查（仅第 0 轮）：
            若答案文本里出现 [来源：片段N] 这类引用标记——
            说明 prompt 明确禁止"没调用工具却标注来源"的情况被违反了，
            大概率是模型跳过了本该调用的 search_knowledge，直接编了答案。
            一旦命中，自动用 tool_choice="search_knowledge" 强制重试一次，
            拿到真实检索结果后再走正常流程，而不是把幻觉答案直接返回给用户。
       从 response.content 中抽取 [N] 引用标记
       返回 AgentResult(answer, steps, citations, processing_stages, trace_id)

若达到 max_iterations 仍未结束：
  强制再发一次不带工具的 generate_chat → 做最终总结
```

这道"安全网"写在 `src/agent/react_loop.py`，是 2026-06 加的——单靠 system prompt
里的规则（见下方"知识库检索可靠性"一节）无法 100% 防住模型跳过检索直接编答案，
所以在代码层面加了一道兜底检测。

---

## 知识库检索流水线

Agent 只会调用一个统一工具 `search_knowledge` 来做知识库 / 网页查询。它是一个很薄的
facade（`agent/tools/search_knowledge.py::SearchKnowledgeTool`）：

1. 总是先调用 `KnowledgeBaseTool.search_typed()`。
2. 只有当 `hit_count == 0`（知识库无命中）或调用方明确传了 `need_fresh_info=True`
   时，才补充调用 `WebSearchTool`。知识库结果和网页结果会分区展示，绝不会混作
   同一事实来源。

真正的检索逻辑在 `agent/tools/search_kb.py::KnowledgeBaseTool.search_typed()` 里，
**不是**简单的"embedding → 向量召回 → rerank"，而是一条专门设计来应对两类
高频问题的多 query、来源感知流水线：

- Excel 测试用例的行，很容易被体量更大的 Bug 列表挤出 top-k；
- "有哪几种方式" / "都能搜索出什么内容" 这类枚举型问题，真实答案往往横跨
  比单次检索 top_k 更多的知识库行，naive 单 query 检索会漏掉一部分。

```
query
  │
  ▼
QueryRewriter（LLM，temperature=0.0）── 枚举型任务，必须保证确定性
  │  生成最多 max_query_variants（默认 5）个改写子查询
  │  · 最多重试 2 次，每次有超时（query_rewriter.timeout_seconds，默认 8s）
  │  · LLM 未启用 / 超时 / 失败 / 只生成 ≤1 个变体
  │      → 回退到规则改写 src/retriever/query_expansion.py::expand_query()
  │  · LLM 改写出的子查询（去掉原始 query 后剩下的部分）会单独保留，
  │      用于后续 rerank —— 这些子查询不带 App 名（"叭嗒"），
  │      避免 rerank 被跨 App 噪声干扰（例如"爱奇艺强拉叭嗒"这类行）
  ▼
┌─ 1. 候选召回（并发）──────────────────────────────────────────────┐
│  · 每个 variant 都跑一次 retrieve_candidates(top_k=max(40, top_k*8)) │
│  · 额外跑一次专门的 Excel boost 召回：                              │
│      retrieve_candidates(                                        │
│          query=sub_queries[0] 或原始 query, top_k=80,             │
│          filters: source_format ∈ {xlsx, xlsm} 且内容不含 "Bug Key:")│
│    —— 没有这一步，体量庞大的 Bug 列表会在进入 rerank 前就把真正的    │
│       测试用例行挤出候选池                                          │
└──────────────────────────────────────────────────────────────────┘
  ▼
2. 稳定去重（Excel 行用 "来源:Sheet:行号" 做 key，其次用向量 id，
   最后用内容 hash 兜底）→ 合并后上限裁到 240 条候选
  ▼
3. 构建来源感知的 rerank 候选池：Excel 候选全部保留，
   Bug / XMind / Other 各裁到 max(top_k, 10) 条——
   Excel 在打分之前绝不能被挤掉
  ▼
4. Rerank（重排打分）
   · 没有 LLM 子查询  → 单次 rerank_candidates(query, pool)
   · 有 LLM 子查询    → 多子查询 rerank：每个子查询各自独立给候选池打分，
     每条候选取所有子查询打分里的最高分，再按子查询轮询交叉排序
     （而不是直接按全局最高分排序）——这一步就是用来防止单个高分模块
     （比如"动画"）把 top_k 名额全占满，挤掉"页面有哪些内容"这类
     枚举问题里其他同样重要的模块（比如"帖子"）
  ▼
5. 来源感知选择最终结果：Excel 优先拿 max(top_k, 5) 个名额，
   剩余名额按 Bug → XMind → Other 的优先级依次补齐
  ▼
6. 按来源优先级排序（Excel > Bug > XMind > Other），分组格式化，
   每条结果标好 [N] 引用编号 → 作为工具的文本结果返回
```

LLM 生成**不**在这一层做——这一层只负责返回排好序、带引用编号的 `SearchResult` 文本，
生成回答是 Agent 层的职责。

### 几个容易踩坑的默认值

| 配置项 | 取值 | 为什么 |
|---|---|---|
| `KnowledgeBaseTool` / `SearchKnowledgeTool` 的默认 `top_k` | **8** | 对齐 `retrieval.final_k`（见下方"配置"一节）。2026-06 之前这个值在三处地方都硬编码成了 5；枚举型问题经常横跨超过 5 条 Excel 行（比如"登录方式"实际有 8 种），即使检索排序完全正确，默认值太小也会把已经召回的正确内容截断丢弃。 |
| `_PER_QUERY_CANDIDATE_K` | 40（或 `top_k*8`） | 每个 query variant 的召回宽度，故意放宽——真正的截断发生在后面第 5 步。 |
| `_EXCEL_BOOST_K` | 80 | 专属 Excel 召回这一路的宽度。 |
| `_MAX_MERGED_CANDIDATES` | 240 | 进入 rerank 候选池之前的硬上限，控制 rerank 延迟。 |
| `query_rewriter` 调用 LLM 时的 temperature | **0.0** | 在 `query_rewriter.py` 里硬编码——改写子查询是枚举型任务，不是创意生成，必须确定性输出。 |

---

## 知识库检索可靠性：已知失效模式与兜底措施

这套流水线在实测中暴露过几类典型问题，记录下来是为了避免以后重新踩坑：

1. **温度（temperature）默认值不够保守**
   全局 `llm.temperature` 默认 0.3，适合通用对话，但答案合成本质上是"事实抽取 + 引用"，
   不是创意生成。`rag chat` CLI（`src/agent/cli.py`）显式把工具调用判断和最终答案合成
   两次 LLM 调用的 temperature 都锁定为 `0.0`（`_CHAT_TEMPERATURE` 常量），不再依赖
   全局默认值——这和 `query_rewriter.py` 自己硬编码 `temperature=0.0` 是同一个取舍。

2. **即使 temperature=0，云端 LLM 也不是完全确定性的**
   DeepSeek 等云端 API 在 temperature=0 下仍可能因浮点非结合性、批处理/MoE 路由等原因
   给出不同结果——这是已知的行业普遍现象，不是本项目代码的 bug，意味着任何纯靠
   prompt 约束的修复都无法保证 100% 生效。

3. **多轮对话中模型会"凭历史跳过检索"**
   随着对话轮数增加，模型有时会判断"前面已经查过类似内容/根据上文已经能回答"，
   于是跳过本该调用的 `search_knowledge`，直接凭训练知识编答案（甚至会编出看起来
   很正式的 `[来源：片段N]` 引用标记）。`prompts/agent.yaml` 里专门加了一个
   "禁止凭历史跳过检索 ⚠️ 最高优先级" 小节，明确规定：只要新一轮问题在问叭嗒
   功能相关内容，永远重新调用 `search_knowledge`，不允许以历史轮次为理由跳过。

4. **代码层兜底：检测"没调用工具却带引用标记"**
   规则 3 单靠 prompt 约束依然有失手概率。`react_loop.py` 因此加了一道确定性检测
   （见上方"ReAct 循环细节"）：本轮第一次 LLM 响应如果没有调用工具，但答案文本里
   却出现了 `[来源：片段N]`，这个组合本身就是被 prompt 明确禁止的伪造信号——
   命中即自动强制重试一次 `search_knowledge`，避免把幻觉答案直接展示给用户。
   局限：如果模型跳过检索但编答案时完全不带引用标记（"裸编"），这层检测目前
   还抓不到——实测中暂未遇到这种情况，但值得留意。

5. **工具默认 `top_k` 和检索引擎配置的预期不一致**
   `retrieval.final_k` 配置的意图是 8，但 `KnowledgeBaseTool` / `SearchKnowledgeTool`
   自己的默认 `top_k` 一直独立硬编码成 5，和配置脱节。已统一改成 8（见上方
   "几个容易踩坑的默认值"）。

---

## LLM Provider

| Provider | 类名 | 特性 |
|----------|-------|---------|
| Anthropic Claude | `ClaudeProvider` | `generate_chat`、`generate_chat_stream`、工具调用 |
| OpenAI / DeepSeek / DashScope | `OpenAIProvider` | 接口相同，`base_url` 可配置 |

消息格式转换在各自 provider 内部完成（`_messages_to_anthropic`、`_messages_to_openai` 等）。

---

## 历史截断

`src/agent/history.py::truncate_history(history, max_tokens, keep_last)`：

1. 无条件保留最后 `keep_last=4` 条消息。
2. 从剩余消息里最旧的那条开始向前回填，只要累计 token 估算（`len(content) / 2.5`）
   不超过 `max_tokens`（默认 4000）。
3. system prompt 和当前用户问题由调用方另外拼接，不算在 `history` 里。

---

## 文档入库

```
DocumentLoader   （PDF / MD / TXT / XLSX / XMind）
      │
      ▼
TextCleaner      （NFKC 归一化、全角转半角、中文引号统一）
      │
      ▼
ChineseChunker   （jieba 分句边界、chunk_size、overlap）
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

## 配置

所有运行时行为都由 `configs/default.yaml` 控制（可按环境覆盖）：

```yaml
agent:
  max_iterations: 10
  max_history_tokens: 4000    # 历史滑窗 token 预算
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
  top_k: 20              # 向量搜索初始召回数
  final_k: 8             # 重排后的最终数量（KnowledgeBaseTool 默认 top_k 与此对齐）
  similarity_threshold: 0.3

llm:
  provider: deepseek
  model: deepseek-chat
  temperature: 0.3       # 全局默认；rag chat CLI 对工具调用/答案合成显式覆盖为 0.0
  max_tokens: 2048
```

---

## 关键设计决策

| # | 决策 | 理由 |
|---|----------|-----------|
| 1 | Agent 主导 | LLM 驱动推理；RAG 只是一个工具，不是固定流水线 |
| 2 | 工具并发执行 | `asyncio.gather`——同一轮里多个工具调用并发跑 |
| 3 | 工具列表配置化 | 在 YAML 里增删工具，不用改代码 |
| 4 | 历史滑窗截断 | 不依赖各 provider 自己的 tokenizer，也能防止 context 溢出 |
| 5 | trace_id 全链路贯穿 | UUID 贯穿所有日志，方便分布式追踪 |
| 6 | 每步 `duration_ms` | 体现在 `AgentStep` 和 `processing_stages` 里，便于观测延迟 |
| 7 | WebSearch 仅作兜底 | 目前是爬取实现，生产环境建议换成稳定的 API |
| 8 | 两阶段检索 | 向量召回（top-20）+ cross-encoder 重排（top-8） |
| 9 | 答案合成用 temperature=0 | 事实抽取/引用任务而非创意生成，`rag chat` CLI 显式覆盖全局默认值 |
| 10 | "跳过检索"伪造信号兜底重试 | prompt 约束 + 代码层检测双保险，应对云端 LLM 在 temperature=0 下仍非完全确定性的现实 |
