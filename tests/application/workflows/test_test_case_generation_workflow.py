"""Test case generation workflow tests."""

from __future__ import annotations

import json

import pytest

from src.application.artifact_repository import LocalArtifactRepository
from src.application.workflows.test_case_generation_workflow import (
    TestCaseGenerationWorkflow,
    default_test_case_nodes,
)
from src.domain.requirements import TestCaseGenerationRequest
from src.llm.types import ChatResponse
from tests.fakes import FakeLLM


class _FakeRetrievalEngine:
    def __init__(self):
        self.calls: list[dict] = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return []


_VALID_IR_JSON = json.dumps({
    "module": "测试模块",
    "summary": "测试需求",
    "actors": [],
    "features": [
        {
            "id": "F001",
            "name": "核心功能",
            "description": "核心功能描述",
            "priority": "P0",
            "acceptance_criteria": ["可以执行主流程"],
            "test_hints": [],
            "dependencies": [],
        }
    ],
    "business_rules": [],
    "state_machines": [],
    "data_entities": [],
    "out_of_scope": [],
})


def _make_workflow_llm(cases: list[dict]) -> FakeLLM:
    return FakeLLM(
        responses=[
            ChatResponse(content=_VALID_IR_JSON, model="fake"),
            ChatResponse(content=json.dumps(cases, ensure_ascii=False), model="fake"),
        ]
    )


def _make_cases_llm(cases: list[dict]) -> FakeLLM:
    return FakeLLM(
        responses=[ChatResponse(content=json.dumps(cases, ensure_ascii=False), model="fake")]
    )


@pytest.mark.asyncio
async def test_workflow_builds_test_design_artifact(tmp_path):
    cases = [
        {
            "title": "正常登录成功",
            "module": "登录",
            "precondition": "用户已注册",
            "steps": "1. 输入账号密码\n2. 点击登录",
            "expected": "进入首页",
            "priority": "P0",
            "type": "正向",
            "notes": "",
        }
    ]
    workflow = TestCaseGenerationWorkflow(
        loader=None,
        cleaner=None,
        retrieval_engine=_FakeRetrievalEngine(),
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
        nodes=default_test_case_nodes(_make_workflow_llm(cases)),
        default_output_dir=str(tmp_path),
    )

    generation = await workflow.run(
        TestCaseGenerationRequest(requirement="用户可以账号密码登录", module="登录")
    )

    assert generation.artifact is not None
    assert generation.artifact.requirement_ir.module == "登录"
    assert generation.artifact.test_points[0].id == "TP001"
    assert generation.artifact.scenarios[0].id == "SC001"
    assert generation.cases[0].title == "正常登录成功"


@pytest.mark.asyncio
async def test_workflow_exports_automation_json_to_directory(tmp_path):
    cases = [
        {
            "title": "展示每日更新",
            "module": "追番表Card",
            "precondition": "本周有追番数据",
            "steps": "1. 打开动画推荐页",
            "expected": "展示每日更新",
            "priority": "P0",
            "type": "正向",
            "ui_display_name": "每日更新",
            "expected_visibility": "展示态",
            "forbidden_locators": ["追番表Card"],
            "notes": "",
        }
    ]
    workflow = TestCaseGenerationWorkflow(
        loader=None,
        cleaner=None,
        retrieval_engine=_FakeRetrievalEngine(),
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
        nodes=default_test_case_nodes(_make_workflow_llm(cases)),
        default_output_dir=str(tmp_path),
    )
    generation = await workflow.run(
        TestCaseGenerationRequest(
            requirement="追番表Card 标题每日更新",
            module="追番表Card",
            generation_mode="automation",
        )
    )

    result = workflow.export_to_directory(generation, str(tmp_path))

    assert result.automation_json_artifact is not None
    payload = json.loads(result.automation_json_artifact.path.read_text(encoding="utf-8"))
    assert payload["cases"][0]["ui_display_name"] == "每日更新"
    assert payload["cases"][0]["forbidden_locators"] == ["追番表Card"]


@pytest.mark.asyncio
async def test_workflow_generates_from_confirmed_analysis_graph(tmp_path):
    cases = [
        {
            "title": "追番表Card正常展示",
            "module": "追番表Card",
            "precondition": "已进入动画频道推荐页",
            "steps": "1. 查看推荐页内容流",
            "expected": "展示追番表Card",
            "priority": "P0",
            "type": "正向",
            "notes": "",
        }
    ]
    graph = {
        "summary": "动画频道新增追番表Card",
        "features": [
            {
                "id": "F001",
                "name": "追番表Card展示",
                "description": "在动画频道推荐页展示追番表Card",
                "priority": "P0",
                "boundaries": ["整周无数据时隐藏模块"],
                "test_focus": ["模块展示位置正确"],
                "dependencies": [],
            }
        ],
        "test_strategy": {"exclusions": ["完整追番表页面"]},
        "_meta": {"analysis_status": "confirmed", "module": "追番表Card"},
    }
    workflow = TestCaseGenerationWorkflow(
        loader=None,
        cleaner=None,
        retrieval_engine=_FakeRetrievalEngine(),
        artifacts=LocalArtifactRepository(base_dir=str(tmp_path)),
        nodes=default_test_case_nodes(_make_cases_llm(cases)),
        default_output_dir=str(tmp_path),
    )

    generation = await workflow.run_from_analysis_graph(
        graph,
        TestCaseGenerationRequest(
            requirement=json.dumps(graph, ensure_ascii=False),
            module="追番表Card",
        ),
    )

    assert generation.artifact is not None
    assert generation.artifact.requirement_ir.features[0].name == "追番表Card展示"
    assert "模块展示位置正确" in generation.artifact.test_points[0].hints
    assert generation.artifact.requirement_ir.out_of_scope == ["完整追番表页面"]
    assert generation.cases[0].title == "追番表Card正常展示"
