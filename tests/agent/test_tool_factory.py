"""Tests for config-driven agent tool factory."""

from src.agent.tool_factory import build_agent_tools
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from tests.fakes import FakeLLM


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


# ── Basic construction ────────────────────────────────────────────────────────

def test_build_agent_tools_default_names(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine)
    assert [t.name for t in tools] == ["knowledge_search", "web_search"]


def test_build_agent_tools_respects_config_order(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["web_search", "knowledge_search"])
    assert [t.name for t in tools] == ["web_search", "knowledge_search"]


def test_build_agent_tools_skips_unknown_names(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["knowledge_search", "ghost_tool"])
    assert [t.name for t in tools] == ["knowledge_search"]


# ── Dict-format tool configs ──────────────────────────────────────────────────

def test_dict_config_overrides_description(fake_embedder, fake_vectordb, fake_reranker):
    """YAML object format: description override is applied to the tool schema."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [
        {
            "name": "knowledge_search",
            "description": "自定义搜索说明",
        }
    ]
    tools = build_agent_tools(engine, config)
    assert len(tools) == 1
    schema = tools[0].to_tool_schema()
    assert schema["description"] == "自定义搜索说明"


def test_dict_config_without_description_uses_default(fake_embedder, fake_vectordb, fake_reranker):
    """When description is omitted in dict config, the code default is preserved."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [{"name": "knowledge_search"}]
    tools = build_agent_tools(engine, config)
    assert len(tools) == 1
    default_desc = tools[0].description
    schema_desc = tools[0].to_tool_schema()["description"]
    assert schema_desc == default_desc


def test_dict_config_mixed_with_string(fake_embedder, fake_vectordb, fake_reranker):
    """String and dict entries can be mixed in the same list."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [
        "knowledge_search",
        {"name": "web_search", "description": "网络搜索工具"},
    ]
    tools = build_agent_tools(engine, config)
    assert [t.name for t in tools] == ["knowledge_search", "web_search"]
    assert tools[1].to_tool_schema()["description"] == "网络搜索工具"


def test_dict_config_system_prompt_passed_to_write_test_cases(
    fake_embedder, fake_vectordb, fake_reranker
):
    """system_prompt in dict config is forwarded to WriteTestCasesTool."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    llm = FakeLLM()
    config = [
        {
            "name": "write_test_cases",
            "system_prompt": "自定义系统提示词",
        }
    ]
    tools = build_agent_tools(engine, config, llm=llm)
    assert len(tools) == 1
    tool = tools[0]
    assert tool._system_prompt == "自定义系统提示词"


def test_dict_config_skips_entry_with_missing_name(fake_embedder, fake_vectordb, fake_reranker):
    """Dict entry without 'name' key is skipped with a warning."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [{"description": "无名工具"}, "knowledge_search"]
    tools = build_agent_tools(engine, config)
    assert [t.name for t in tools] == ["knowledge_search"]


def test_write_test_cases_skipped_without_llm(fake_embedder, fake_vectordb, fake_reranker):
    """write_test_cases is skipped (not crashed) when no LLM is provided."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["write_test_cases"], llm=None)
    assert tools == []


# ── Description override (BaseTool) ──────────────────────────────────────────

def test_override_description_affects_schema(fake_embedder, fake_vectordb, fake_reranker):
    """override_description() updates to_tool_schema() but not description property."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["knowledge_search"])
    tool = tools[0]
    original = tool.description

    tool.override_description("全新说明")
    assert tool.to_tool_schema()["description"] == "全新说明"
    assert tool.description == original  # code default unchanged


def test_override_description_empty_string_reverts(fake_embedder, fake_vectordb, fake_reranker):
    """override_description('') is treated as no override (falls back to default)."""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["knowledge_search"])
    tool = tools[0]
    original = tool.description

    tool.override_description("")
    assert tool.to_tool_schema()["description"] == original
