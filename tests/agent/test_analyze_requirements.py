"""Tests for AnalyzeRequirementsTool and its integration with tool_factory."""

from __future__ import annotations

import json

import pytest

from src.agent.tool_factory import build_agent_tools
from src.agent.tools.analyze_requirements import (
    AnalyzeRequirementsTool,
    _normalise_graph,
    _save_json,
    _save_markdown,
)
from src.llm.types import ChatResponse
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from tests.fakes import FakeLLM

# ── Shared fixture data ───────────────────────────────────────────────────────

_MINIMAL_REQ = "用户可以通过账号和密码登录系统。"

_VALID_GRAPH_JSON = json.dumps({
    "summary": "账号密码登录功能",
    "actors": ["注册用户"],
    "features": [
        {
            "id": "F001",
            "name": "账号密码登录",
            "description": "用户输入账号和密码完成登录",
            "priority": "P0",
            "risk_level": "high",
            "risk_reason": "涉及安全认证",
            "boundaries": ["密码长度6-20位", "连续错误5次锁定"],
            "test_focus": ["正常登录", "密码错误"],
            "dependencies": [],
        }
    ],
    "state_transitions": [
        {
            "entity": "登录状态",
            "states": ["未登录", "已登录", "已锁定"],
            "transitions": [
                {"from": "未登录", "to": "已登录", "trigger": "认证成功", "condition": ""},
                {"from": "未登录", "to": "已锁定", "trigger": "连续5次失败", "condition": ""},
            ],
        }
    ],
    "risks": [
        {
            "area": "安全认证",
            "level": "high",
            "description": "密码暴力破解风险",
            "suggestion": "测试账号锁定机制",
        },
        {
            "area": "会话管理",
            "level": "medium",
            "description": "Token 过期处理",
            "suggestion": "测试 token 过期后的跳转",
        },
    ],
    "clarifications": [
        {
            "id": "Q001",
            "question": "账号锁定后如何解锁？",
            "context": "需求未说明解锁方式",
            "impact": "影响反向用例设计",
        }
    ],
    "test_strategy": {
        "scope": "登录模块全量测试",
        "focus_areas": ["安全认证", "错误提示"],
        "exclusions": [],
        "suggestion": "优先测试安全相关场景",
    },
})


@pytest.fixture
def llm_with_valid_graph() -> FakeLLM:
    """FakeLLM that returns a valid RequirementGraph JSON."""
    return FakeLLM(response_text=_VALID_GRAPH_JSON)


@pytest.fixture
def tool(llm_with_valid_graph, tmp_path) -> AnalyzeRequirementsTool:
    return AnalyzeRequirementsTool(
        llm=llm_with_valid_graph,
        output_dir=str(tmp_path),
    )


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


# ── Schema / metadata ─────────────────────────────────────────────────────────

def test_tool_name(tool):
    assert tool.name == "analyze_requirements"


def test_tool_schema_has_required_requirement(tool):
    params = tool.parameters
    assert "requirement" in params["required"]
    assert "kb_context" in params["properties"]
    assert "module" in params["properties"]


# ── Successful execution ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_creates_json_and_md_files(tool, tmp_path):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    json_files = list(tmp_path.glob("*_req_graph.json"))
    md_files = list(tmp_path.glob("*_analysis.md"))
    assert len(json_files) == 1, "Expected one JSON output"
    assert len(md_files) == 1, "Expected one Markdown output"
    assert "JSON 文件" in result
    assert "Markdown 报告" in result


@pytest.mark.asyncio
async def test_execute_json_contains_correct_structure(tool, tmp_path):
    await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    json_file = next(tmp_path.glob("*_req_graph.json"))
    with open(json_file, encoding="utf-8") as f:
        graph = json.load(f)
    assert graph["summary"] == "账号密码登录功能"
    assert len(graph["features"]) == 1
    assert graph["features"][0]["id"] == "F001"
    assert len(graph["state_transitions"]) == 1
    assert len(graph["risks"]) == 2
    assert len(graph["clarifications"]) == 1


@pytest.mark.asyncio
async def test_execute_json_has_meta_section(tool, tmp_path):
    await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    json_file = next(tmp_path.glob("*_req_graph.json"))
    with open(json_file, encoding="utf-8") as f:
        graph = json.load(f)
    assert "_meta" in graph
    assert graph["_meta"]["module"] == "登录"
    assert "generated_at" in graph["_meta"]
    assert graph["_meta"]["has_kb_context"] is False


@pytest.mark.asyncio
async def test_execute_kb_context_reflected_in_meta(tool, tmp_path):
    await tool.execute(
        requirement=_MINIMAL_REQ,
        module="登录",
        kb_context="现有登录测试用例：正常登录、密码错误",
    )
    json_file = next(tmp_path.glob("*_req_graph.json"))
    with open(json_file, encoding="utf-8") as f:
        graph = json.load(f)
    assert graph["_meta"]["has_kb_context"] is True


@pytest.mark.asyncio
async def test_execute_module_appears_in_filename(tool, tmp_path):
    await tool.execute(requirement=_MINIMAL_REQ, module="用户登录")
    json_files = list(tmp_path.glob("用户登录_*_req_graph.json"))
    assert len(json_files) == 1


@pytest.mark.asyncio
async def test_execute_result_contains_feature_and_risk_counts(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert "功能点" in result
    assert "风险点" in result
    assert "待澄清问题" in result


@pytest.mark.asyncio
async def test_execute_result_highlights_high_risks(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert "安全认证" in result  # high risk area name


@pytest.mark.asyncio
async def test_execute_custom_output_dir(tmp_path, llm_with_valid_graph):
    custom_dir = tmp_path / "custom" / "sub"
    tool = AnalyzeRequirementsTool(llm=llm_with_valid_graph, output_dir=str(tmp_path))
    await tool.execute(requirement=_MINIMAL_REQ, output_dir=str(custom_dir))
    assert custom_dir.exists()
    assert len(list(custom_dir.glob("*_req_graph.json"))) == 1


# ── LLM prompt inspection ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kb_context_included_in_llm_prompt(tool, llm_with_valid_graph):
    await tool.execute(
        requirement=_MINIMAL_REQ,
        kb_context="历史用例：密码错误返回错误提示",
    )
    last_msgs = llm_with_valid_graph.last_messages
    user_msg = next(m for m in last_msgs if m.role == "user")
    assert "历史用例" in user_msg.content


@pytest.mark.asyncio
async def test_system_prompt_override_applied(tmp_path):
    llm = FakeLLM(response_text=_VALID_GRAPH_JSON)
    tool = AnalyzeRequirementsTool(
        llm=llm,
        system_prompt="自定义系统提示词",
        output_dir=str(tmp_path),
    )
    await tool.execute(requirement=_MINIMAL_REQ)
    sys_msg = next(m for m in llm.last_messages if m.role == "system")
    assert sys_msg.content == "自定义系统提示词"


# ── JSON parsing robustness ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_strips_markdown_fences(tmp_path):
    wrapped = f"```json\n{_VALID_GRAPH_JSON}\n```"
    llm = FakeLLM(response_text=wrapped)
    tool = AnalyzeRequirementsTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "JSON 文件" in result


@pytest.mark.asyncio
async def test_execute_handles_json_embedded_in_text(tmp_path):
    text = f"Here is the graph:\n{_VALID_GRAPH_JSON}\nDone."
    llm = FakeLLM(response_text=text)
    tool = AnalyzeRequirementsTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "JSON 文件" in result


@pytest.mark.asyncio
async def test_execute_returns_error_on_invalid_json(tmp_path):
    llm = FakeLLM(response_text="完全不是 JSON 内容")
    tool = AnalyzeRequirementsTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "未能生成" in result or "错误" in result


# ── Input validation ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_empty_requirement_returns_error(tool):
    result = await tool.execute(requirement="")
    assert "错误" in result


@pytest.mark.asyncio
async def test_execute_whitespace_requirement_returns_error(tool):
    result = await tool.execute(requirement="   \n  ")
    assert "错误" in result


# ── _normalise_graph ──────────────────────────────────────────────────────────

def test_normalise_graph_fills_missing_keys():
    graph = _normalise_graph({"summary": "测试"}, "登录")
    assert graph["actors"] == []
    assert graph["features"] == []
    assert graph["risks"] == []
    assert graph["clarifications"] == []
    assert "test_strategy" in graph


def test_normalise_graph_keeps_existing_values():
    raw = json.loads(_VALID_GRAPH_JSON)
    graph = _normalise_graph(raw, "登录")
    assert graph["summary"] == "账号密码登录功能"
    assert len(graph["features"]) == 1


def test_normalise_graph_default_summary_when_empty():
    graph = _normalise_graph({}, "登录模块")
    assert "登录模块" in graph["summary"]


# ── File helpers ──────────────────────────────────────────────────────────────

def test_save_json_valid_utf8(tmp_path):
    graph = json.loads(_VALID_GRAPH_JSON)
    path = tmp_path / "graph.json"
    _save_json(graph, path)
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["summary"] == graph["summary"]


def test_save_markdown_contains_key_sections(tmp_path):
    graph = json.loads(_VALID_GRAPH_JSON)
    graph = _normalise_graph(graph, "登录")
    path = tmp_path / "report.md"
    _save_markdown(graph, path, "登录")
    text = path.read_text(encoding="utf-8")
    assert "# 需求分析报告" in text
    assert "## 功能点清单" in text
    assert "## 风险分析" in text
    assert "## 待澄清问题" in text
    assert "## 测试策略建议" in text


def test_save_markdown_includes_state_transitions(tmp_path):
    graph = json.loads(_VALID_GRAPH_JSON)
    graph = _normalise_graph(graph, "登录")
    path = tmp_path / "report.md"
    _save_markdown(graph, path, "登录")
    text = path.read_text(encoding="utf-8")
    assert "登录状态" in text
    assert "认证成功" in text


# ── tool_factory integration ──────────────────────────────────────────────────

def test_factory_registers_analyze_requirements(
    fake_embedder, fake_vectordb, fake_reranker
):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    llm = FakeLLM()
    tools = build_agent_tools(engine, ["analyze_requirements"], llm=llm)
    assert len(tools) == 1
    assert tools[0].name == "analyze_requirements"


def test_factory_skips_analyze_requirements_without_llm(
    fake_embedder, fake_vectordb, fake_reranker
):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["analyze_requirements"], llm=None)
    assert tools == []


def test_factory_system_prompt_forwarded(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    llm = FakeLLM()
    config = [{"name": "analyze_requirements", "system_prompt": "自定义提示词"}]
    tools = build_agent_tools(engine, config, llm=llm)
    assert tools[0]._system_prompt == "自定义提示词"


def test_factory_description_override(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    llm = FakeLLM()
    config = [{"name": "analyze_requirements", "description": "自定义工具说明"}]
    tools = build_agent_tools(engine, config, llm=llm)
    assert tools[0].to_tool_schema()["description"] == "自定义工具说明"


# ── Agent orchestration ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_calls_analyze_requirements(
    fake_embedder, fake_vectordb, fake_reranker
):
    """Smoke test: Agent can route to analyze_requirements and return a result."""
    from src.agent.react_loop import ReActAgent
    from src.llm.types import ToolCall

    tool_call_resp = ChatResponse(
        content="",
        model="fake",
        stop_reason="tool_use",
        usage={},
        tool_calls=[
            ToolCall(
                id="t1",
                name="analyze_requirements",
                arguments={"requirement": _MINIMAL_REQ, "module": "登录"},
            )
        ],
    )
    final_resp = ChatResponse(
        content="需求分析已完成，报告已保存。",
        model="fake",
        stop_reason="end_turn",
        usage={},
    )

    # Third response needed if Agent reflects on tool output
    llm = FakeLLM(
        responses=[tool_call_resp, final_resp, final_resp],
    )
    analyze_tool = AnalyzeRequirementsTool(
        llm=FakeLLM(response_text=_VALID_GRAPH_JSON),
        output_dir="/tmp/test_outputs",
    )

    agent = ReActAgent(
        llm=llm,
        tools=[analyze_tool],
        system_prompt="你是测试助手。",
        max_iterations=5,
    )
    result = await agent.run(query="分析登录功能需求")
    assert result.answer
