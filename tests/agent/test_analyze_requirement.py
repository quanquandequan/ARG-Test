"""AnalyzeRequirementTool 测试。"""

from __future__ import annotations

import json

import pytest

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH
from src.agent.tool_result import ToolExecutionResult
from src.agent.tools.analyze_requirement import AnalyzeRequirementTool
from src.llm.types import ChatResponse
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import SearchResult
from tests.fakes import FakeLLM

_VALID_IR_JSON = json.dumps({
    "module": "登录",
    "summary": "账号密码登录",
    "actors": [],
    "features": [
        {
            "id": "F001",
            "name": "账号密码登录",
            "description": "用户输入账号密码完成登录",
            "priority": "P0",
            "acceptance_criteria": ["登录成功进入首页"],
            "test_hints": [],
            "dependencies": [],
        }
    ],
    "business_rules": [],
    "state_machines": [],
    "data_entities": [],
    "out_of_scope": [],
})

_VALID_REVIEW_JSON = json.dumps({
    "overall_quality": "needs_clarification",
    "score": 72,
    "ambiguities": [
        {
            "id": "A001",
            "location": "F001",
            "description": "登录失败文案未明确",
            "suggestion": "补充错误提示文案",
        }
    ],
    "gaps": [
        {
            "id": "G001",
            "description": "锁定时长未明确",
            "impact": "影响异常用例设计",
            "question": "账号锁定时长是多少？",
        }
    ],
    "risks": [
        {
            "area": "安全",
            "level": "high",
            "description": "暴力破解风险",
            "mitigation": "验证错误次数与锁定机制",
        }
    ],
    "suggestions": ["补充错误提示文案"],
})

_VALID_GRAPH_JSON = json.dumps({
    "summary": "登录需求分析",
    "actors": ["注册用户"],
    "features": [
        {
            "id": "F001",
            "name": "账号密码登录",
            "description": "用户输入账号密码完成登录",
            "priority": "P0",
            "risk_level": "high",
            "risk_reason": "安全认证",
            "boundaries": ["密码错误5次锁定"],
            "test_focus": ["正常登录", "错误锁定"],
            "dependencies": [],
        }
    ],
    "state_transitions": [],
    "risks": [
        {
            "area": "安全认证",
            "level": "high",
            "description": "密码暴力破解风险",
            "suggestion": "重点测试锁定机制",
        }
    ],
    "clarifications": [
        {
            "id": "Q001",
            "question": "锁定后如何解锁？",
            "context": "需求未说明",
            "impact": "影响异常恢复测试",
        }
    ],
    "test_strategy": {
        "scope": "登录模块",
        "focus_areas": ["安全认证"],
        "exclusions": [],
        "suggestion": "优先覆盖安全相关场景",
    },
})


def test_analyze_requirement_uses_passthrough_final_answer(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
):
    tool = AnalyzeRequirementTool(
        llm=FakeLLM(),
        retrieval_engine=_make_engine(fake_embedder, fake_vectordb, fake_reranker),
    )

    assert tool.final_answer_mode == FINAL_ANSWER_PASSTHROUGH


class _FakeRetrievalEngine:
    def __init__(self, candidates: list[SearchResult] | None = None):
        self.candidates = candidates or []
        self.retrieve_calls = 0

    async def search(self, **kwargs):
        return []

    async def retrieve_candidates(self, **kwargs):
        self.retrieve_calls += 1
        return list(self.candidates)

    async def rerank_candidates(self, query, candidates, top_k=None):
        if top_k is None:
            return list(candidates)
        return list(candidates[:top_k])


class _RecordingLLM(FakeLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_calls = []

    async def generate_chat(self, messages, **kwargs):
        self.message_calls.append(messages)
        return await super().generate_chat(messages, **kwargs)


def _user_message(messages) -> str:
    return next(message.content for message in messages if message.role == "user")


def _make_engine(embedder, vectordb, reranker) -> RetrievalEngine:
    dense = DenseRetriever(embedder, vectordb)
    return RetrievalEngine(dense_retriever=dense, reranker=reranker)


@pytest.mark.asyncio
async def test_analyze_requirement_draft_does_not_create_json_artifacts(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
    tmp_path,
):
    llm = FakeLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=_VALID_REVIEW_JSON, model="fake"),
            ChatResponse(content=_VALID_GRAPH_JSON, model="fake"),
        ]
    )
    tool = AnalyzeRequirementTool(
        llm=llm,
        retrieval_engine=_make_engine(fake_embedder, fake_vectordb, fake_reranker),
    )

    result = await tool.execute(
        requirement="用户可以使用账号密码登录，密码错误5次后锁定账号。",
        module="登录",
        output_dir=str(tmp_path),
    )

    assert "需求分析草稿（待确认）" in result
    assert "草稿阶段未生成最终 JSON" in result
    assert "质量评分：72/100" in result
    assert len(list(tmp_path.glob("*_ir.json"))) == 0
    assert len(list(tmp_path.glob("*_review.json"))) == 0
    assert len(list(tmp_path.glob("*_req_graph.json"))) == 0


@pytest.mark.asyncio
async def test_analyze_requirement_final_creates_confirmed_artifacts(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
    tmp_path,
):
    llm = FakeLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=_VALID_REVIEW_JSON, model="fake"),
            ChatResponse(content=_VALID_GRAPH_JSON, model="fake"),
        ]
    )
    tool = AnalyzeRequirementTool(
        llm=llm,
        retrieval_engine=_make_engine(fake_embedder, fake_vectordb, fake_reranker),
    )

    result = await tool.execute_typed(
        requirement="用户可以使用账号密码登录，密码错误5次后锁定账号。",
        module="登录",
        output_dir=str(tmp_path),
        analysis_mode="final",
        clarification_answers="错误提示文案为：账号或密码错误。",
    )

    assert "确认版需求分析完成" in result.content
    assert "RequirementIR" in result.content
    assert "评审报告" in result.content
    assert "分析结果" in result.content
    assert result.metadata["analysis_status"] == "confirmed"
    assert len(list(tmp_path.glob("*_ir.json"))) == 1
    assert len(list(tmp_path.glob("*_review.json"))) == 1
    json_files = list(tmp_path.glob("*_req_graph.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["_meta"]["analysis_status"] == "confirmed"
    assert payload["_meta"]["clarification_answers_used"] is True


@pytest.mark.asyncio
async def test_analyze_requirement_reads_requirement_file(tmp_path):
    req_file = tmp_path / "req.md"
    req_file.write_text(
        "# PRD\n新增追番表 Card，点击加追后按钮变为已加追。",
        encoding="utf-8",
    )
    llm = _RecordingLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=_VALID_REVIEW_JSON, model="fake"),
            ChatResponse(content=_VALID_GRAPH_JSON, model="fake"),
        ]
    )
    tool = AnalyzeRequirementTool(llm=llm, retrieval_engine=_FakeRetrievalEngine())

    result = await tool.execute_typed(
        requirement_file=str(req_file),
        module="追番表Card",
        output_dir=str(tmp_path),
    )

    parser_user = _user_message(llm.message_calls[0])
    assert "新增追番表 Card" in parser_user
    assert str(req_file) not in parser_user
    assert result.metadata["requirement_source_path"] == str(req_file.resolve())


@pytest.mark.asyncio
async def test_analyze_requirement_extracts_file_path_from_short_command(tmp_path):
    req_file = tmp_path / "req.md"
    req_file.write_text("动画频道推荐页新增每日更新模块。", encoding="utf-8")
    llm = _RecordingLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=_VALID_REVIEW_JSON, model="fake"),
            ChatResponse(content=_VALID_GRAPH_JSON, model="fake"),
        ]
    )
    tool = AnalyzeRequirementTool(llm=llm, retrieval_engine=_FakeRetrievalEngine())

    await tool.execute_typed(
        requirement=f"请读取文件 {req_file} 的内容并分析",
        module="追番表Card",
        output_dir=str(tmp_path),
    )

    parser_user = _user_message(llm.message_calls[0])
    assert "动画频道推荐页新增每日更新模块" in parser_user
    assert "请读取文件" not in parser_user


@pytest.mark.asyncio
async def test_analyze_requirement_file_error_happens_before_retrieval(tmp_path):
    engine = _FakeRetrievalEngine()
    llm = FakeLLM()
    tool = AnalyzeRequirementTool(llm=llm, retrieval_engine=engine)

    result = await tool.execute_typed(
        requirement_file=str(tmp_path / "missing.md"),
        module="追番表Card",
    )

    assert "无法读取需求文件" in result.content
    assert engine.retrieve_calls == 0


@pytest.mark.asyncio
async def test_analyze_requirement_final_requires_clarification_answers(tmp_path):
    tool = AnalyzeRequirementTool(llm=FakeLLM(), retrieval_engine=_FakeRetrievalEngine())

    result = await tool.execute_typed(
        requirement="用户可以使用账号密码登录。",
        module="登录",
        output_dir=str(tmp_path),
        analysis_mode="final",
    )

    assert "final 模式需要提供 clarification_answers" in result.content
    assert len(list(tmp_path.glob("*_req_graph.json"))) == 0


@pytest.mark.asyncio
async def test_parser_ignores_kb_and_analyzer_receives_auxiliary_context(tmp_path):
    kb_candidate = SearchResult(
        id="kb-1",
        document_id="ACN_cases",
        content="历史追番表页面支持分页加载和漫画/动画切换。",
        score=0.91,
        metadata={"source_path": "/kb/ACN_cases.xlsx", "format": "xlsx"},
    )
    llm = _RecordingLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=_VALID_REVIEW_JSON, model="fake"),
            ChatResponse(content=_VALID_GRAPH_JSON, model="fake"),
        ]
    )
    tool = AnalyzeRequirementTool(
        llm=llm,
        retrieval_engine=_FakeRetrievalEngine(candidates=[kb_candidate]),
    )

    result = await tool.execute_typed(
        requirement="当前需求：动画频道推荐页新增追番表 Card，支持星期切换和加追。",
        module="追番表Card",
        output_dir=str(tmp_path),
    )

    parser_user = _user_message(llm.message_calls[0])
    analyzer_user = _user_message(llm.message_calls[2])
    assert "历史知识库参考" not in parser_user
    assert "历史追番表页面支持分页加载" in analyzer_user
    assert "不得作为当前需求事实来源" in analyzer_user
    assert "需求分析草稿（待确认）" in result.content
    assert result.metadata["analysis_status"] == "draft"


@pytest.mark.asyncio
async def test_analyze_requirement_fails_when_reviewer_has_no_data(tmp_path):
    llm = FakeLLM(responses=[ChatResponse(content=_VALID_IR_JSON, model="fake")])
    tool = AnalyzeRequirementTool(llm=llm, retrieval_engine=_FakeRetrievalEngine())

    class BrokenReviewer:
        async def execute_typed(self, **kwargs):
            return ToolExecutionResult(content="评审 JSON 解析失败。")

    tool._reviewer_tool = BrokenReviewer()

    result = await tool.execute_typed(
        requirement="用户可以使用账号密码登录。",
        module="登录",
        output_dir=str(tmp_path),
        request_id="req-1",
    )

    assert result.data is None
    assert "需求分析失败：requirement_reviewer" in result.content
    assert "需求分析完成" not in result.content
    assert result.metadata["request_id"] == "req-1"
    assert result.metadata["failed_stage"] == "requirement_reviewer"


@pytest.mark.asyncio
async def test_analyze_requirement_fails_when_analyzer_has_no_data(tmp_path):
    llm = FakeLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=_VALID_REVIEW_JSON, model="fake"),
        ]
    )
    tool = AnalyzeRequirementTool(llm=llm, retrieval_engine=_FakeRetrievalEngine())

    class BrokenAnalyzer:
        async def execute_typed(self, **kwargs):
            return ToolExecutionResult(content="分析 JSON 解析失败。")

    tool._analyzer_tool = BrokenAnalyzer()

    result = await tool.execute_typed(
        requirement="用户可以使用账号密码登录。",
        module="登录",
        output_dir=str(tmp_path),
        request_id="req-2",
    )

    assert result.data is None
    assert "需求分析失败：analyze_requirements" in result.content
    assert "需求分析完成" not in result.content
    assert result.metadata["failed_stage"] == "analyze_requirements"
