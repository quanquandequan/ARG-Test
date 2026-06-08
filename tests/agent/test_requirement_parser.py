"""RequirementParserTool 测试。"""

from __future__ import annotations

import json

import pytest

from src.agent.tool_factory import build_agent_tools
from src.agent.tools.requirement_parser import RequirementParserTool, _render_markdown
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from src.services.requirement_ir import RequirementIR
from tests.fakes import FakeLLM

_MINIMAL_REQ = "用户可以通过账号和密码登录系统，密码错误5次后账号锁定。"

_VALID_IR_JSON = json.dumps({
    "module": "用户登录",
    "summary": "账号密码登录，含错误锁定",
    "actors": [{"name": "注册用户", "role": "执行登录"}],
    "features": [
        {
            "id": "F001",
            "name": "账号密码登录",
            "description": "用户输入凭证完成登录",
            "priority": "P0",
            "acceptance_criteria": ["成功后跳转首页", "密码错误显示提示"],
            "test_hints": ["边界：密码长度6/20位", "连续5次错误锁定"],
            "dependencies": [],
        }
    ],
    "business_rules": [
        {
            "id": "R001",
            "description": "账号锁定",
            "condition": "IF 密码连续错误5次",
            "outcome": "THEN 账号锁定30分钟",
            "related_features": ["F001"],
        }
    ],
    "state_machines": [],
    "data_entities": [],
    "out_of_scope": [],
})


@pytest.fixture
def llm_valid() -> FakeLLM:
    return FakeLLM(response_text=_VALID_IR_JSON)


@pytest.fixture
def tool(llm_valid, tmp_path) -> RequirementParserTool:
    return RequirementParserTool(llm=llm_valid, output_dir=str(tmp_path))


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


# ── Schema ────────────────────────────────────────────────────────────────────

def test_tool_name(tool):
    assert tool.name == "requirement_parser"


def test_parameters_has_required_requirement(tool):
    assert "requirement" in tool.parameters["required"]
    assert "ir_file" not in tool.parameters.get("required", [])


# ── Successful execution ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_creates_json_and_md_files(tool, tmp_path):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert len(list(tmp_path.glob("*_ir.json"))) == 1
    assert len(list(tmp_path.glob("*_ir_summary.md"))) == 1
    assert "IR 文件" in result


@pytest.mark.asyncio
async def test_json_validates_as_requirement_ir(tool, tmp_path):
    await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    json_file = next(tmp_path.glob("*_ir.json"))
    ir = RequirementIR.model_validate_json(json_file.read_text(encoding="utf-8"))
    assert ir.module == "登录"
    assert ir.feature_count() == 1
    assert ir.features[0].id == "F001"
    assert ir.source_length == len(_MINIMAL_REQ)
    assert ir.has_kb_context is False


@pytest.mark.asyncio
async def test_kb_context_reflected_in_meta(tool, tmp_path):
    await tool.execute(
        requirement=_MINIMAL_REQ, module="登录", kb_context="现有测试用例..."
    )
    json_file = next(tmp_path.glob("*_ir.json"))
    ir = RequirementIR.model_validate_json(json_file.read_text(encoding="utf-8"))
    assert ir.has_kb_context is True


@pytest.mark.asyncio
async def test_result_contains_ir_file_marker(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert "[IR_FILE=" in result


@pytest.mark.asyncio
async def test_result_lists_features(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert "F001" in result
    assert "账号密码登录" in result


@pytest.mark.asyncio
async def test_module_in_filename(tool, tmp_path):
    await tool.execute(requirement=_MINIMAL_REQ, module="用户登录")
    files = list(tmp_path.glob("用户登录_*_ir.json"))
    assert len(files) == 1


@pytest.mark.asyncio
async def test_custom_output_dir(tmp_path, llm_valid):
    custom = tmp_path / "custom"
    tool = RequirementParserTool(llm=llm_valid, output_dir=str(tmp_path))
    await tool.execute(requirement=_MINIMAL_REQ, output_dir=str(custom))
    assert custom.exists()
    assert len(list(custom.glob("*_ir.json"))) == 1


# ── LLM prompt inspection ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kb_context_in_llm_prompt(tool, llm_valid):
    await tool.execute(requirement=_MINIMAL_REQ, kb_context="历史用例：...")
    user_msg = next(m for m in llm_valid.last_messages if m.role == "user")
    assert "历史用例" in user_msg.content
    assert user_msg.content.index("需求文档：") < user_msg.content.index(
        "【历史知识库参考（辅助）】"
    )


@pytest.mark.asyncio
async def test_parser_prompt_declares_prd_as_only_fact_source(tool, llm_valid):
    await tool.execute(requirement=_MINIMAL_REQ, kb_context="历史用例：...")
    sys_msg = next(m for m in llm_valid.last_messages if m.role == "system")
    user_msg = next(m for m in llm_valid.last_messages if m.role == "user")

    assert "当前输入的需求文档是唯一的需求事实来源" in sys_msg.content
    assert "features、acceptance_criteria、business_rules" in sys_msg.content
    assert "不得作为当前需求事实来源" in user_msg.content


@pytest.mark.asyncio
async def test_system_prompt_override(tmp_path):
    llm = FakeLLM(response_text=_VALID_IR_JSON)
    tool = RequirementParserTool(llm=llm, system_prompt="自定义", output_dir=str(tmp_path))
    await tool.execute(requirement=_MINIMAL_REQ)
    sys_msg = next(m for m in llm.last_messages if m.role == "system")
    assert sys_msg.content == "自定义"


# ── Robustness ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strips_markdown_fences(tmp_path):
    llm = FakeLLM(response_text=f"```json\n{_VALID_IR_JSON}\n```")
    tool = RequirementParserTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "IR 文件" in result


@pytest.mark.asyncio
async def test_invalid_json_returns_error(tmp_path):
    llm = FakeLLM(response_text="这不是JSON")
    tool = RequirementParserTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "未能" in result or "错误" in result


@pytest.mark.asyncio
async def test_empty_requirement_returns_error(tool):
    result = await tool.execute(requirement="")
    assert "错误" in result


# ── _render_markdown ──────────────────────────────────────────────────────────

def test_render_markdown_has_required_sections(tmp_path):
    ir = RequirementIR.model_validate_json(_VALID_IR_JSON)
    ir = ir.model_copy(update={"module": "登录"})
    md = _render_markdown(ir)
    assert "# RequirementIR 摘要" in md
    assert "## 功能点" in md
    assert "## 业务规则" in md


# ── tool_factory integration ──────────────────────────────────────────────────

def test_factory_registers_requirement_parser(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    llm = FakeLLM()
    tools = build_agent_tools(engine, ["requirement_parser"], llm=llm)
    assert len(tools) == 1
    assert tools[0].name == "requirement_parser"


def test_factory_skips_without_llm(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["requirement_parser"], llm=None)
    assert tools == []


def test_factory_system_prompt_forwarded(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    cfg = [{"name": "requirement_parser", "system_prompt": "自定义提示"}]
    tools = build_agent_tools(engine, cfg, llm=FakeLLM())
    assert tools[0]._system_prompt == "自定义提示"
