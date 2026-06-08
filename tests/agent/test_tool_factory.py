"""配置驱动 Agent 工具工厂测试。"""

import pytest

from src.agent.tool_factory import build_agent_tools
from src.application.artifact_repository import LocalArtifactRepository
from src.application.execution_service import MobileExecutionService
from src.application.requirement_services import TestCaseGenerationService
from src.application.workflows.test_case_generation_workflow import (
    TestCaseGenerationWorkflow,
)
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from tests.fakes import FakeLLM


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


class _FakeRetrievalEngine:
    async def search(self, **kwargs):
        return []


class _FakeExecutionWorkflow:
    async def execute(self, request):
        return request


def _make_service() -> TestCaseGenerationService:
    workflow = TestCaseGenerationWorkflow(
        loader=None,
        cleaner=None,
        retrieval_engine=_FakeRetrievalEngine(),
        artifacts=LocalArtifactRepository(base_dir="./outputs"),
        nodes=[],
    )
    return TestCaseGenerationService(workflow=workflow)


def _make_mobile_execution_service() -> MobileExecutionService:
    return MobileExecutionService(workflow=_FakeExecutionWorkflow())


# ── Basic construction ────────────────────────────────────────────────────────

def test_build_agent_tools_default_names(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(
        engine,
        llm=FakeLLM(),
        test_case_generation_service=_make_service(),
        mobile_execution_service=_make_mobile_execution_service(),
    )
    assert [t.name for t in tools] == [
        "search_knowledge",
        "analyze_requirement",
        "design_test_cases",
        "execute_scenario",
    ]


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
    """YAML 对象格式：description 覆盖会应用到工具 schema。"""
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
    """dict 配置省略 description 时，保留代码默认值。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [{"name": "knowledge_search"}]
    tools = build_agent_tools(engine, config)
    assert len(tools) == 1
    default_desc = tools[0].description
    schema_desc = tools[0].to_tool_schema()["description"]
    assert schema_desc == default_desc


def test_dict_config_mixed_with_string(fake_embedder, fake_vectordb, fake_reranker):
    """字符串和 dict 条目可以混在同一个列表中。"""
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
    """dict 配置中的 system_prompt 会传给 WriteTestCasesTool。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    service = _make_service()
    config = [
        {
            "name": "write_test_cases",
            "system_prompt": "自定义系统提示词",
        }
    ]
    tools = build_agent_tools(
        engine,
        config,
        llm=FakeLLM(),
        test_case_generation_service=service,
    )
    assert len(tools) == 1
    tool = tools[0]
    assert tool._system_prompt == "自定义系统提示词"


def test_analyze_requirement_rejects_system_prompt_override(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [
        {
            "name": "analyze_requirement",
            "system_prompt": "不应被静默丢弃",
        }
    ]

    with pytest.raises(ValueError, match="不支持 system_prompt 覆盖"):
        build_agent_tools(engine, config, llm=FakeLLM())


def test_dict_config_skips_entry_with_missing_name(fake_embedder, fake_vectordb, fake_reranker):
    """缺少 'name' 键的 dict 条目会被跳过并记录警告。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    config = [{"description": "无名工具"}, "knowledge_search"]
    tools = build_agent_tools(engine, config)
    assert [t.name for t in tools] == ["knowledge_search"]


def test_write_test_cases_skipped_without_service(fake_embedder, fake_vectordb, fake_reranker):
    """未提供 Service 时会跳过 write_test_cases，而不是崩溃。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["write_test_cases"], llm=FakeLLM())
    assert tools == []


def test_mobile_tools_share_injected_runtime(fake_embedder, fake_vectordb, fake_reranker):
    from src.services.page_cache import PageCache
    from tests.mobile.conftest import FakeAppiumDriverManager

    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    driver = FakeAppiumDriverManager()
    page_cache = PageCache()

    tools = build_agent_tools(
        engine,
        ["device_tool", "screen_tool", "action_tool", "assertion_tool"],
        driver_manager=driver,
        page_cache=page_cache,
    )

    assert tools[0]._mgr is driver
    assert tools[1]._mgr is driver
    assert tools[1]._cache is page_cache
    assert tools[2]._mgr is driver
    assert tools[2]._cache is page_cache
    assert tools[3]._mgr is driver


# ── Description override (BaseTool) ──────────────────────────────────────────

def test_override_description_affects_schema(fake_embedder, fake_vectordb, fake_reranker):
    """override_description() 会更新 to_tool_schema()，但不改变 description 属性。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["knowledge_search"])
    tool = tools[0]
    original = tool.description

    tool.override_description("全新说明")
    assert tool.to_tool_schema()["description"] == "全新说明"
    assert tool.description == original  # 代码默认值保持不变


def test_override_description_empty_string_reverts(fake_embedder, fake_vectordb, fake_reranker):
    """override_description('') 视为无覆盖，并回退到默认值。"""
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["knowledge_search"])
    tool = tools[0]
    original = tool.description

    tool.override_description("")
    assert tool.to_tool_schema()["description"] == original
