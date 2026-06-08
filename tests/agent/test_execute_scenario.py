"""ExecuteScenarioTool 与 ExecutionWorkflow 测试。"""

from __future__ import annotations

import json

import pytest

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH
from src.agent.tool_factory import build_agent_tools
from src.agent.tools.execute_scenario import ExecuteScenarioTool
from src.application.artifact_repository import LocalArtifactRepository
from src.application.execution_service import MobileExecutionService
from src.application.workflows.execution_workflow import ExecutionWorkflow
from src.services.page_cache import PageCache
from tests.mobile.conftest import FakeAppiumDriverManager


def _make_tool(tmp_path, driver: FakeAppiumDriverManager) -> ExecuteScenarioTool:
    workflow = ExecutionWorkflow(
        driver_manager=driver,
        page_cache=PageCache(ttl_seconds=30.0),
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
    )
    service = MobileExecutionService(workflow=workflow)
    return ExecuteScenarioTool(service=service)


def test_execute_scenario_uses_passthrough_final_answer(tmp_path):
    tool = _make_tool(tmp_path, FakeAppiumDriverManager())

    assert tool.final_answer_mode == FINAL_ANSWER_PASSTHROUGH


@pytest.mark.asyncio
async def test_execute_scenario_happy_path(tmp_path):
    payload = {
        "module": "登录",
        "cases": [
            {
                "id": "LOGIN-001",
                "title": "登录页展示登录按钮",
                "automation_steps": [
                    {"tool": "device_tool", "action": "launch_app"},
                    {"tool": "screen_tool", "action": "get_current_screen"},
                ],
                "assertions": [
                    {"tool": "assertion_tool", "action": "assert_text", "text": "登录"}
                ],
            }
        ],
    }
    json_path = tmp_path / "login_cases.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tool = _make_tool(tmp_path, FakeAppiumDriverManager())

    result = await tool.execute(
        automation_json_path=str(json_path),
        case_id="LOGIN-001",
        app_package="com.example.app",
    )

    assert "执行结果：PASS" in result
    assert list((tmp_path / "executions").glob("*_pass.json"))
    assert list((tmp_path / "executions").glob("*_pass.png"))


@pytest.mark.asyncio
async def test_execute_scenario_assertion_failure(tmp_path):
    payload = {
        "module": "登录",
        "cases": [
            {
                "id": "LOGIN-002",
                "title": "错误断言示例",
                "automation_steps": [
                    {"tool": "screen_tool", "action": "get_current_screen"},
                ],
                "assertions": [
                    {"tool": "assertion_tool", "action": "assert_text", "text": "首页"}
                ],
            }
        ],
    }
    json_path = tmp_path / "login_cases_fail.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tool = _make_tool(tmp_path, FakeAppiumDriverManager())

    result = await tool.execute(
        automation_json_path=str(json_path),
        case_id="LOGIN-002",
    )

    assert "执行结果：FAIL" in result
    assert "失败原因" in result
    assert list((tmp_path / "executions").glob("*_fail.json"))


def test_execute_scenario_and_debug_tool_can_share_driver(
    fake_embedder,
    fake_vectordb,
    fake_reranker,
    tmp_path,
):
    from src.retriever.dense_retriever import DenseRetriever
    from src.retriever.retrieval_engine import RetrievalEngine

    driver = FakeAppiumDriverManager()
    page_cache = PageCache(ttl_seconds=30.0)
    workflow = ExecutionWorkflow(
        driver_manager=driver,
        page_cache=page_cache,
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
    )
    service = MobileExecutionService(workflow=workflow)
    engine = RetrievalEngine(
        dense_retriever=DenseRetriever(fake_embedder, fake_vectordb),
        reranker=fake_reranker,
    )

    tools = build_agent_tools(
        engine,
        ["device_tool", "execute_scenario"],
        mobile_execution_service=service,
        driver_manager=driver,
        page_cache=page_cache,
    )

    device_tool, execute_tool = tools
    assert device_tool._mgr is driver
    assert execute_tool._service._workflow._driver_manager is driver
    assert execute_tool._service._workflow._screen_tool._cache is page_cache
