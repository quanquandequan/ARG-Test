"""Tests for WriteTestCasesTool — requirements-to-Excel test case generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.tools.write_test_cases import WriteTestCasesTool
from src.llm.types import ChatResponse
from tests.fakes import FakeLLM

# ── Fixture helpers ───────────────────────────────────────────────────────────

_SAMPLE_CASES = [
    {
        "title": "正常登录成功",
        "module": "登录",
        "precondition": "用户已注册",
        "steps": "1. 打开 App\n2. 输入正确账号密码\n3. 点击登录",
        "expected": "登录成功，跳转首页",
        "priority": "P0",
        "type": "正向",
        "notes": "",
    },
    {
        "title": "密码错误登录失败",
        "module": "登录",
        "precondition": "用户已注册",
        "steps": "1. 输入正确账号\n2. 输入错误密码\n3. 点击登录",
        "expected": "提示'密码错误'，停留在登录页",
        "priority": "P0",
        "type": "反向",
        "notes": "",
    },
]

_REQUIREMENT = "用户可以使用账号和密码登录应用。连续失败 5 次需要锁定账号。"


def _make_llm_with_cases(cases: list[dict] | None = None) -> FakeLLM:
    payload = json.dumps(cases or _SAMPLE_CASES, ensure_ascii=False)
    return FakeLLM(response_text=payload)


@pytest.fixture
def tmp_output(tmp_path) -> Path:
    return tmp_path / "test_cases"


@pytest.fixture
def tool(tmp_output) -> WriteTestCasesTool:
    return WriteTestCasesTool(llm=_make_llm_with_cases(), output_dir=str(tmp_output))


# ── Schema / metadata ─────────────────────────────────────────────────────────

def test_tool_name(tool):
    assert tool.name == "write_test_cases"


def test_tool_schema_has_required_requirement(tool):
    schema = tool.to_tool_schema()
    assert "requirement" in schema["parameters"]["required"]
    assert "kb_samples" in schema["parameters"]["properties"]
    assert "module" in schema["parameters"]["properties"]
    assert "output_dir" in schema["parameters"]["properties"]


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_execute_creates_excel_file(tool, tmp_output):
    result = await tool.execute(requirement=_REQUIREMENT, module="登录")
    assert "已生成" in result
    assert ".xlsx" in result
    xlsx_files = list(tmp_output.glob("*.xlsx"))
    assert len(xlsx_files) == 1


async def test_excel_file_has_correct_row_count(tool, tmp_output):
    await tool.execute(requirement=_REQUIREMENT, module="登录")
    from openpyxl import load_workbook
    wb = load_workbook(list(tmp_output.glob("*.xlsx"))[0])
    ws = wb.active
    # Row 1 = header; remaining rows = cases
    data_rows = ws.max_row - 1
    assert data_rows == len(_SAMPLE_CASES)


async def test_excel_file_has_headers(tool, tmp_output):
    await tool.execute(requirement=_REQUIREMENT, module="登录")
    from openpyxl import load_workbook
    wb = load_workbook(list(tmp_output.glob("*.xlsx"))[0])
    ws = wb.active
    headers = [ws.cell(1, col).value for col in range(1, ws.max_column + 1)]
    assert "用例标题" in headers
    assert "预期结果" in headers
    assert "优先级" in headers


async def test_execute_returns_case_count_in_summary(tool):
    result = await tool.execute(requirement=_REQUIREMENT, module="登录")
    assert f"{len(_SAMPLE_CASES)} 条" in result


async def test_execute_uses_module_in_filename(tool, tmp_output):
    await tool.execute(requirement=_REQUIREMENT, module="登录")
    xlsx_files = list(tmp_output.glob("*.xlsx"))
    assert any("登录" in f.name for f in xlsx_files)


async def test_execute_custom_output_dir(tmp_path):
    custom_dir = tmp_path / "custom_output"
    llm = _make_llm_with_cases()
    tool = WriteTestCasesTool(llm=llm, output_dir=str(tmp_path / "default"))
    result = await tool.execute(
        requirement=_REQUIREMENT,
        output_dir=str(custom_dir),
    )
    assert ".xlsx" in result
    assert list(custom_dir.glob("*.xlsx")), "Excel should be in custom output dir"


async def test_execute_passes_kb_samples_to_llm(tmp_path):
    llm = _make_llm_with_cases()
    tool = WriteTestCasesTool(llm=llm, output_dir=str(tmp_path))
    kb_sample = "样本用例：正向登录测试"
    await tool.execute(requirement=_REQUIREMENT, kb_samples=kb_sample)
    user_msg = next(m for m in llm.last_messages if m.role == "user")
    assert kb_sample in user_msg.content


# ── JSON parsing robustness ───────────────────────────────────────────────────

async def test_execute_strips_markdown_fences(tmp_path):
    cases_json = json.dumps(_SAMPLE_CASES, ensure_ascii=False)
    llm = FakeLLM(response_text=f"```json\n{cases_json}\n```")
    tool = WriteTestCasesTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_REQUIREMENT)
    assert ".xlsx" in result


async def test_execute_handles_json_embedded_in_text(tmp_path):
    cases_json = json.dumps([_SAMPLE_CASES[0]], ensure_ascii=False)
    llm = FakeLLM(response_text=f"以下是生成的用例：\n{cases_json}\n希望对你有帮助。")
    tool = WriteTestCasesTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_REQUIREMENT)
    assert ".xlsx" in result


async def test_execute_normalises_missing_fields(tmp_path):
    minimal_cases = [{"title": "最小用例"}]
    llm = FakeLLM(response_text=json.dumps(minimal_cases, ensure_ascii=False))
    tool = WriteTestCasesTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_REQUIREMENT, module="测试")
    assert ".xlsx" in result


# ── Edge / error cases ────────────────────────────────────────────────────────

async def test_execute_empty_requirement_returns_error(tool):
    result = await tool.execute(requirement="")
    assert "错误" in result


async def test_execute_whitespace_requirement_returns_error(tool):
    result = await tool.execute(requirement="   ")
    assert "错误" in result


async def test_execute_invalid_json_returns_error(tmp_path):
    llm = FakeLLM(response_text="这不是有效的 JSON，也不包含数组。")
    tool = WriteTestCasesTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_REQUIREMENT)
    assert "LLM 未能生成" in result or "错误" in result


# ── tool_factory integration ──────────────────────────────────────────────────

def test_factory_registers_write_test_cases(
    fake_embedder, fake_vectordb, fake_reranker
):
    from src.agent.tool_factory import build_agent_tools
    from src.retriever.dense_retriever import DenseRetriever
    from src.retriever.retrieval_engine import RetrievalEngine

    dense = DenseRetriever(fake_embedder, fake_vectordb)
    engine = RetrievalEngine(dense_retriever=dense, reranker=fake_reranker)
    llm = FakeLLM()

    tools = build_agent_tools(engine, ["write_test_cases"], llm=llm)
    assert len(tools) == 1
    assert tools[0].name == "write_test_cases"


def test_factory_skips_write_test_cases_without_llm(
    fake_embedder, fake_vectordb, fake_reranker
):
    from src.agent.tool_factory import build_agent_tools
    from src.retriever.dense_retriever import DenseRetriever
    from src.retriever.retrieval_engine import RetrievalEngine

    dense = DenseRetriever(fake_embedder, fake_vectordb)
    engine = RetrievalEngine(dense_retriever=dense, reranker=fake_reranker)

    tools = build_agent_tools(engine, ["write_test_cases"], llm=None)
    assert tools == []


# ── Agent integration ─────────────────────────────────────────────────────────

async def test_agent_orchestrates_kb_search_then_write(
    fake_embedder, fake_vectordb, fake_reranker, tmp_path
):
    """Agent 先调用 knowledge_search，再调用 write_test_cases，返回文件路径。"""
    from src.agent.react_loop import ReActAgent
    from src.agent.tools.search_kb import KnowledgeBaseTool
    from src.retriever.dense_retriever import DenseRetriever
    from src.retriever.retrieval_engine import RetrievalEngine
    from src.llm.types import ToolCall

    dense = DenseRetriever(fake_embedder, fake_vectordb)
    engine = RetrievalEngine(dense_retriever=dense, reranker=fake_reranker)
    kb_tool = KnowledgeBaseTool(engine)

    cases_json = json.dumps(_SAMPLE_CASES, ensure_ascii=False)
    write_tool = WriteTestCasesTool(
        llm=FakeLLM(response_text=cases_json),
        output_dir=str(tmp_path),
    )

    expected_path = str(tmp_path / "登录_20260101.xlsx")
    agent_responses = [
        # Step 1: call knowledge_search
        ChatResponse(
            content="",
            model="fake",
            stop_reason="tool_use",
            usage={},
            tool_calls=[ToolCall(
                id="tc1", name="knowledge_search",
                arguments={"query": "测试用例 格式"},
            )],
        ),
        # Step 2: call write_test_cases
        ChatResponse(
            content="",
            model="fake",
            stop_reason="tool_use",
            usage={},
            tool_calls=[ToolCall(
                id="tc2", name="write_test_cases",
                arguments={"requirement": _REQUIREMENT, "module": "登录"},
            )],
        ),
        # Step 3: final answer
        ChatResponse(
            content=f"已生成测试用例文件：\n路径：{expected_path}\n用例数量：2 条",
            model="fake",
            stop_reason="end_turn",
            usage={},
        ),
    ]
    agent_llm = FakeLLM(responses=agent_responses)
    agent = ReActAgent(
        llm=agent_llm,
        tools=[kb_tool, write_tool],
        system_prompt="你是测试助手。",
    )

    result = await agent.run("请为登录功能生成测试用例")
    assert result.iterations == 3
    assert "路径" in result.answer or "已生成" in result.answer
