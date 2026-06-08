"""Test design domain model tests."""

from src.domain.artifacts.test_design_artifact import TestDesignArtifact
from src.domain.requirement import Feature, RequirementIR
from src.domain.requirements import GeneratedTestCase
from src.domain.test_design import (
    ExecutionAction,
    ExecutionPlan,
    TestPoint,
    TestScenario,
)
from src.services.requirement_ir import RequirementIR as CompatRequirementIR


def test_requirement_ir_compat_reexport_points_to_domain_model():
    assert CompatRequirementIR is RequirementIR


def test_test_design_artifact_collects_core_assets():
    ir = RequirementIR(
        module="登录",
        summary="登录功能",
        features=[Feature(id="F001", name="登录", description="账号密码登录")],
    )
    point = TestPoint(id="TP001", title="登录成功", feature_id="F001")
    plan = ExecutionPlan(
        actions=[ExecutionAction(action="tap", target="登录按钮")],
        assertions=["进入首页"],
    )
    scenario = TestScenario(
        id="SC001",
        title="用户成功登录",
        point_id=point.id,
        execution_intent=plan,
    )
    case = GeneratedTestCase(
        title="正常登录",
        module="登录",
        precondition="用户已注册",
        steps="1. 输入账号密码\n2. 点击登录",
        expected="进入首页",
        priority="P0",
        case_type="正向",
    )

    artifact = TestDesignArtifact(
        module="登录",
        generation_mode="manual",
        requirement_ir=ir,
        test_points=[point],
        scenarios=[scenario],
        test_cases=[case],
    )

    assert artifact.case_count == 1
    assert artifact.scenarios[0].execution_intent.actions[0].target == "登录按钮"
