"""Shared exporter helpers."""

from __future__ import annotations

import json

from src.domain.requirements import GeneratedTestCase

DEFAULT_COLUMNS = [
    ("用例编号", 12),
    ("所属模块", 14),
    ("用例标题", 30),
    ("前置条件", 22),
    ("测试步骤", 38),
    ("预期结果", 30),
    ("优先级", 8),
    ("用例类型", 10),
    ("备注", 16),
]

AUTOMATION_COLUMNS = [
    ("用例编号", 12),
    ("所属模块", 14),
    ("用例标题", 30),
    ("前置条件", 22),
    ("自动化数据准备", 26),
    ("业务名称", 18),
    ("界面展示名", 18),
    ("页面路径", 28),
    ("期望可见性", 14),
    ("定位链", 38),
    ("锚点文本", 20),
    ("搜索策略", 34),
    ("禁用定位词", 28),
    ("测试步骤", 38),
    ("预期结果", 30),
    ("优先级", 8),
    ("用例类型", 10),
    ("选择器", 34),
    ("自动化步骤", 46),
    ("自动化断言", 38),
    ("备注", 16),
]


def normalise_generation_mode(value: str = "") -> str:
    value = (value or "manual").strip().lower()
    return "automation" if value in {"automation", "auto", "ui", "mobile"} else "manual"


def normalise_expected_visibility(value) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "show": "visible",
        "display": "visible",
        "displayed": "visible",
        "展示": "visible",
        "展示态": "visible",
        "可见": "visible",
        "visible": "visible",
        "empty": "empty",
        "empty_state": "empty",
        "空态": "empty",
        "无数据": "empty",
        "hidden": "hidden",
        "hide": "hidden",
        "隐藏": "hidden",
        "隐藏态": "hidden",
        "blocked": "blocked",
        "block": "blocked",
        "阻塞": "blocked",
    }
    return mapping.get(text, text if text in {"visible", "empty", "hidden", "blocked"} else "")


def maybe_json(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    if text[0] not in "[{":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def stringify_extra(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def case_to_dict(case: GeneratedTestCase | dict) -> dict:
    if isinstance(case, GeneratedTestCase):
        return {
            "title": case.title,
            "module": case.module,
            "precondition": case.precondition,
            "steps": case.steps,
            "expected": case.expected,
            "priority": case.priority,
            "type": case.case_type,
            "notes": case.notes,
            "data_setup": case.data_setup,
            "business_name": case.business_name,
            "ui_display_name": case.ui_display_name,
            "page_route": case.page_route,
            "locator_chain": case.locator_chain,
            "anchor_text": case.anchor_text,
            "search_strategy": case.search_strategy,
            "expected_visibility": case.expected_visibility,
            "forbidden_locators": case.forbidden_locators,
            "selectors": case.selectors,
            "automation_steps": case.automation_steps,
            "assertions": case.assertions,
        }
    return dict(case)
