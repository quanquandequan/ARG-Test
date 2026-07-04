"""需求流程 CLI 辅助函数与直達路由测试。"""

from __future__ import annotations

from src.agent.base_tool import BaseTool
from src.agent.requirement_flow import (
    RequirementFlowSession,
    RequirementGoal,
    build_cli_design_cases_query,
    build_cli_final_query,
    detect_requirement_goal,
    extract_analysis_json_path,
    format_clarification_answers,
    is_draft_pending_confirmation,
    parse_clarification_questions,
    parse_cli_design_cases_payload,
    parse_cli_final_payload,
)
from src.agent.react_loop import (
    ReActAgent,
    _build_cli_design_cases_tool_call,
    _build_cli_final_tool_call,
)
from src.llm.base import BaseLLM
from src.llm.types import ChatResponse


class _StubLLM(BaseLLM):
    async def generate_chat(self, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None):
        return ChatResponse(content="stub", stop_reason="end_turn")

    async def generate_chat_stream(self, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None):
        yield ChatResponse(content="stub", stop_reason="end_turn")


class _EchoTool(BaseTool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "echo"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return "ok"


def test_detect_requirement_goal():
    assert detect_requirement_goal("请对这份需求做需求分析") == RequirementGoal.ANALYSIS_ONLY
    assert detect_requirement_goal("根据 PRD 编写测试用例 /path/a.md") == RequirementGoal.MANUAL_CASES
    assert detect_requirement_goal("输出自动化 case ./req.md") == RequirementGoal.AUTOMATION_CASES
    assert detect_requirement_goal("叭嗒有哪些模块") is None


def test_parse_clarification_questions():
    draft = """
需求分析草稿（待确认）：追番表

需求确认问题：
请按编号逐条回答，回答完成后我会基于你的确认生成 confirmed 需求分析 JSON。
  1. 空态时展示什么文案？（来源：G001）
  2. 点击作品跳转哪里？（来源：G002）

草稿功能点：3 个
"""
    questions = parse_clarification_questions(draft)
    assert len(questions) == 2
    assert questions[0].index == 1
    assert "空态" in questions[0].text
    assert questions[0].source_id == "G001"


def test_format_clarification_answers():
    text = format_clarification_answers([
        (1, "空态文案？", "展示暂无更新"),
        (2, "跳转目标？", "动画详情页"),
    ])
    assert "1. 空态文案？" in text
    assert "答：展示暂无更新" in text


def test_is_draft_pending_confirmation():
    assert is_draft_pending_confirmation("需求分析草稿（待确认）：模块A")
    assert not is_draft_pending_confirmation("确认版需求分析完成")


def test_extract_analysis_json_path():
    answer = "确认版需求分析完成：模块\n分析结果：./outputs/requirements/foo_req_graph.json\n"
    assert extract_analysis_json_path(answer).endswith("foo_req_graph.json")


def test_cli_marker_payload_roundtrip():
    session = RequirementFlowSession(
        goal=RequirementGoal.MANUAL_CASES,
        draft_args={"requirement_file": "/tmp/a.md", "module": "追番表"},
    )
    query = build_cli_final_query(session, "1. Q\n答：A")
    payload = parse_cli_final_payload(query)
    assert payload is not None
    assert payload["requirement_file"] == "/tmp/a.md"
    assert payload["goal"] == "manual_cases"

    design_query = build_cli_design_cases_query("/tmp/x_req_graph.json", "manual", "追番表")
    design_payload = parse_cli_design_cases_payload(design_query)
    assert design_payload is not None
    assert design_payload["generation_mode"] == "manual"


def test_build_cli_final_tool_call():
    query = build_cli_final_query(
        RequirementFlowSession(
            goal=RequirementGoal.ANALYSIS_ONLY,
            draft_args={"requirement": "PRD text", "module": "M"},
        ),
        "确认",
    )
    tc = _build_cli_final_tool_call(query)
    assert tc is not None
    assert tc.name == "analyze_requirement"
    assert tc.arguments["analysis_mode"] == "final"
    assert tc.arguments["clarification_answers"] == "确认"


def test_build_cli_design_cases_tool_call():
    query = build_cli_design_cases_query("/out/a_req_graph.json", "automation")
    tc = _build_cli_design_cases_tool_call(query)
    assert tc is not None
    assert tc.name == "design_test_cases"
    assert tc.arguments["generation_mode"] == "automation"


def test_react_agent_direct_cli_final_route():
    agent = ReActAgent(
        llm=_StubLLM(),
        tools=[_EchoTool("analyze_requirement")],
        system_prompt="test",
    )
    query = build_cli_final_query(
        RequirementFlowSession(
            goal=RequirementGoal.ANALYSIS_ONLY,
            draft_args={"requirement": "x"},
        ),
        "答：确认",
    )
    tc = agent._build_direct_tool_call(query)
    assert tc is not None
    assert tc.name == "analyze_requirement"
