"""Tests for RequirementReviewerTool."""

from __future__ import annotations

import json

import pytest

from src.agent.tool_factory import build_agent_tools
from src.agent.tools.requirement_reviewer import RequirementReviewerTool
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from tests.fakes import FakeLLM

_MINIMAL_REQ = "用户可以通过账号和密码登录，密码错误5次锁定。"

_VALID_REVIEW_JSON = json.dumps({
    "overall_quality": "needs_clarification",
    "score": 65,
    "ambiguities": [
        {
            "id": "A001",
            "location": "F001",
            "description": "密码规则未明确",
            "suggestion": "补充密码长度和字符集要求",
        }
    ],
    "gaps": [
        {
            "id": "G001",
            "description": "解锁方式未说明",
            "impact": "无法设计解锁测试",
            "question": "账号锁定后如何解锁？",
        }
    ],
    "risks": [
        {
            "area": "安全",
            "level": "high",
            "description": "暴力破解风险",
            "mitigation": "重点测试锁定机制",
        }
    ],
    "suggestions": ["补充密码规则", "明确解锁流程"],
})

_VALID_IR_JSON = json.dumps({
    "module": "用户登录",
    "summary": "账号密码登录",
    "actors": [],
    "features": [
        {
            "id": "F001",
            "name": "登录",
            "description": "账号密码登录",
            "priority": "P0",
            "acceptance_criteria": ["登录成功跳转首页"],
            "test_hints": [],
            "dependencies": [],
        }
    ],
    "business_rules": [],
    "state_machines": [],
    "data_entities": [],
    "out_of_scope": [],
})


@pytest.fixture
def llm_valid() -> FakeLLM:
    return FakeLLM(response_text=_VALID_REVIEW_JSON)


@pytest.fixture
def tool(llm_valid, tmp_path) -> RequirementReviewerTool:
    return RequirementReviewerTool(llm=llm_valid, output_dir=str(tmp_path))


@pytest.fixture
def ir_file(tmp_path) -> str:
    path = tmp_path / "test_ir.json"
    path.write_text(_VALID_IR_JSON, encoding="utf-8")
    return str(path)


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


# ── Schema ────────────────────────────────────────────────────────────────────

def test_tool_name(tool):
    assert tool.name == "requirement_reviewer"


# ── Successful execution — raw requirement ────────────────────────────────────

@pytest.mark.asyncio
async def test_raw_requirement_creates_files(tool, tmp_path):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert len(list(tmp_path.glob("*_review.json"))) == 1
    assert len(list(tmp_path.glob("*_review.md"))) == 1
    assert "评审完成" in result


@pytest.mark.asyncio
async def test_result_contains_score(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert "65" in result


@pytest.mark.asyncio
async def test_result_shows_ambiguities(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    assert "密码规则" in result


@pytest.mark.asyncio
async def test_result_shows_quality_gate_hint(tool):
    result = await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    # Score = 65, below 70 → should warn
    assert "澄清" in result or "⚠️" in result


# ── Execution from IR file ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ir_file_input(tool, ir_file, tmp_path):
    result = await tool.execute(ir_file=ir_file)
    assert "评审完成" in result
    assert len(list(tmp_path.glob("*_review.json"))) == 1


@pytest.mark.asyncio
async def test_ir_file_module_extracted_from_ir(tool, ir_file):
    result = await tool.execute(ir_file=ir_file)
    # Module should come from the IR ("用户登录")
    assert "用户登录" in result


@pytest.mark.asyncio
async def test_ir_file_not_found_raises(tool):
    with pytest.raises(FileNotFoundError):
        await tool.execute(ir_file="/nonexistent/path.json")


# ── ReviewResult saved correctly ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_saved_json_validates_as_review_result(tool, tmp_path):
    from src.services.requirement_ir import ReviewResult

    await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    json_file = next(tmp_path.glob("*_review.json"))
    review = ReviewResult.model_validate_json(json_file.read_text(encoding="utf-8"))
    assert review.score == 65
    assert len(review.ambiguities) == 1
    assert len(review.gaps) == 1


# ── Robustness ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strips_markdown_fences(tmp_path):
    llm = FakeLLM(response_text=f"```json\n{_VALID_REVIEW_JSON}\n```")
    tool = RequirementReviewerTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "评审完成" in result


@pytest.mark.asyncio
async def test_invalid_json_returns_error(tmp_path):
    llm = FakeLLM(response_text="这不是JSON")
    tool = RequirementReviewerTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "未能" in result or "错误" in result


@pytest.mark.asyncio
async def test_no_input_returns_error(tool):
    result = await tool.execute()
    assert "错误" in result


# ── Markdown report ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_markdown_contains_required_sections(tool, tmp_path):
    await tool.execute(requirement=_MINIMAL_REQ, module="登录")
    md_file = next(tmp_path.glob("*_review.md"))
    text = md_file.read_text(encoding="utf-8")
    assert "## 评审结论" in text
    assert "## 歧义问题" in text
    assert "## 信息缺口" in text
    assert "## 风险评估" in text


# ── Quality gate hint ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_high_score_shows_proceed_hint(tmp_path):
    good_review = json.dumps(
        {**json.loads(_VALID_REVIEW_JSON), "score": 90, "overall_quality": "good"}
    )
    llm = FakeLLM(response_text=good_review)
    tool = RequirementReviewerTool(llm=llm, output_dir=str(tmp_path))
    result = await tool.execute(requirement=_MINIMAL_REQ)
    assert "✅" in result or "test_point_generator" in result


# ── tool_factory integration ──────────────────────────────────────────────────

def test_factory_registers_reviewer(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["requirement_reviewer"], llm=FakeLLM())
    assert len(tools) == 1
    assert tools[0].name == "requirement_reviewer"


def test_factory_skips_without_llm(fake_embedder, fake_vectordb, fake_reranker):
    engine = _make_engine(fake_embedder, fake_vectordb, fake_reranker)
    tools = build_agent_tools(engine, ["requirement_reviewer"], llm=None)
    assert tools == []
