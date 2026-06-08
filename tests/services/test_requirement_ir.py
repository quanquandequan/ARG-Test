"""RequirementIR 与 ReviewResult schema / 辅助方法的单元测试。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.services.requirement_ir import (
    RequirementIR,
    ReviewResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _minimal_ir_dict(**overrides) -> dict:
    base = {"module": "登录", "summary": "账号密码登录"}
    return {**base, **overrides}


def _full_ir_dict() -> dict:
    return {
        "module": "登录",
        "summary": "账号密码登录功能，支持错误锁定",
        "actors": [{"name": "注册用户", "role": "执行登录操作"}],
        "features": [
            {
                "id": "F001",
                "name": "账号密码登录",
                "description": "用户输入账号和密码完成登录",
                "priority": "P0",
                "acceptance_criteria": ["登录成功后跳转首页", "密码错误时显示错误提示"],
                "test_hints": ["边界：密码长度6/20位", "异常：连续5次错误锁定"],
                "dependencies": [],
            }
        ],
        "business_rules": [
            {
                "id": "R001",
                "description": "账号锁定规则",
                "condition": "IF 密码连续错误5次",
                "outcome": "THEN 账号锁定30分钟",
                "related_features": ["F001"],
            }
        ],
        "state_machines": [
            {
                "entity": "登录状态",
                "states": ["未登录", "已登录", "已锁定"],
                "initial_state": "未登录",
                "transitions": [
                    {
                        "from_state": "未登录",
                        "to_state": "已登录",
                        "trigger": "认证成功",
                        "guard": "",
                    }
                ],
            }
        ],
        "data_entities": [
            {
                "name": "登录请求",
                "fields": [
                    {
                        "name": "username",
                        "field_type": "string",
                        "constraints": ["max_length=50", "required"],
                        "required": True,
                    }
                ],
            }
        ],
        "out_of_scope": ["第三方 OAuth 登录"],
    }


# ── RequirementIR validation ──────────────────────────────────────────────────

def test_minimal_ir_is_valid():
    ir = RequirementIR(**_minimal_ir_dict())
    assert ir.module == "登录"
    assert ir.features == []
    assert ir.version == "1.0"


def test_full_ir_parses_correctly():
    ir = RequirementIR.model_validate(_full_ir_dict())
    assert ir.feature_count() == 1
    assert ir.features[0].id == "F001"
    assert ir.features[0].priority == "P0"
    assert len(ir.business_rules) == 1
    assert len(ir.state_machines) == 1
    assert len(ir.data_entities) == 1


def test_p0_features_filter():
    ir = RequirementIR.model_validate(_full_ir_dict())
    assert len(ir.p0_features()) == 1
    assert ir.p0_features()[0].id == "F001"


def test_invalid_priority_raises():
    data = _full_ir_dict()
    data["features"][0]["priority"] = "P9"
    with pytest.raises(ValidationError):
        RequirementIR.model_validate(data)


def test_compact_summary_contains_counts():
    ir = RequirementIR.model_validate(_full_ir_dict())
    summary = ir.to_compact_summary()
    assert "1" in summary   # 功能数量
    assert "功能点" in summary
    assert "业务规则" in summary


def test_ir_json_round_trip():
    ir = RequirementIR.model_validate(_full_ir_dict())
    serialised = ir.model_dump_json()
    restored = RequirementIR.model_validate_json(serialised)
    assert restored.module == ir.module
    assert restored.feature_count() == ir.feature_count()


# ── RequirementIR.from_llm_json ───────────────────────────────────────────────

def test_from_llm_json_parses_plain_json():
    ir = RequirementIR.from_llm_json(json.dumps(_full_ir_dict()))
    assert ir is not None
    assert ir.module == "登录"


def test_from_llm_json_strips_markdown_fences():
    wrapped = f"```json\n{json.dumps(_full_ir_dict())}\n```"
    ir = RequirementIR.from_llm_json(wrapped)
    assert ir is not None


def test_from_llm_json_extracts_embedded_json():
    text = f"Here is the IR:\n{json.dumps(_minimal_ir_dict())}\nDone."
    ir = RequirementIR.from_llm_json(text)
    assert ir is not None
    assert ir.module == "登录"


def test_from_llm_json_returns_none_on_invalid():
    assert RequirementIR.from_llm_json("not JSON at all") is None
    assert RequirementIR.from_llm_json("[]") is None   # 数组而非对象


# ── ReviewResult ──────────────────────────────────────────────────────────────

def _review_dict() -> dict:
    return {
        "overall_quality": "needs_clarification",
        "score": 65,
        "ambiguities": [
            {
                "id": "A001",
                "location": "F001",
                "description": "密码规则未说明",
                "suggestion": "明确密码长度和字符集要求",
            }
        ],
        "gaps": [
            {
                "id": "G001",
                "description": "未说明账号锁定解锁方式",
                "impact": "无法设计解锁测试用例",
                "question": "账号锁定30分钟后自动解锁还是需要人工介入？",
            }
        ],
        "risks": [
            {
                "area": "安全认证",
                "level": "high",
                "description": "暴力破解风险",
                "mitigation": "重点测试锁定机制",
            }
        ],
        "suggestions": ["补充密码规则说明", "确认解锁流程"],
    }


def test_review_result_parses_correctly():
    review = ReviewResult.model_validate(_review_dict())
    assert review.score == 65
    assert review.overall_quality == "needs_clarification"
    assert len(review.ambiguities) == 1
    assert len(review.gaps) == 1
    assert len(review.risks) == 1


def test_review_invalid_quality_raises():
    data = _review_dict()
    data["overall_quality"] = "unknown"
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(data)


def test_review_score_out_of_range_raises():
    data = _review_dict()
    data["score"] = 150
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(data)


def test_review_compact_summary():
    review = ReviewResult.model_validate(_review_dict())
    summary = review.to_compact_summary()
    assert "65" in summary
    assert "歧义" in summary


def test_review_from_llm_json():
    review = ReviewResult.from_llm_json(json.dumps(_review_dict()))
    assert review is not None
    assert review.score == 65


def test_review_from_llm_json_strips_fences():
    wrapped = f"```json\n{json.dumps(_review_dict())}\n```"
    review = ReviewResult.from_llm_json(wrapped)
    assert review is not None
