"""Generated test case view generator node."""

from __future__ import annotations

import ast
import json
import re

from src.core.logging import get_logger
from src.core.prompt_loader import require_prompt_fields
from src.domain.requirements import GeneratedTestCase
from src.llm.base import BaseLLM
from src.llm.types import Message
from src.services.exporters.common import (
    normalise_expected_visibility,
    stringify_extra,
)
from src.services.workflow_base import WorkflowContext, WorkflowNode

logger = get_logger(__name__)


class CaseGeneratorNode(WorkflowNode):
    """Generate exportable test cases from scenarios and requirement context."""

    def __init__(
        self,
        llm: BaseLLM,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        from src.core.config import get_config

        cfg = get_config().get("test_generator", {})
        self._llm = llm
        self._temperature = (
            temperature
            if temperature is not None
            else float(cfg.get("temperature", 0.3))
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None else int(cfg.get("max_tokens", 8192))
        )
        prompt = require_prompt_fields(
            "case_generator",
            [
                "system_prompt",
                "user_template",
                "automation_requirements",
                "automation_batch_template",
                "json_repair_system_prompt",
                "json_repair_user_template",
                "kb_section_template",
            ],
        )
        self._system_prompt = (
            system_prompt
            or cfg.get("system_prompt", "")
            or prompt["system_prompt"]
        )
        self._user_template = prompt["user_template"]
        self._automation_requirements = prompt["automation_requirements"]
        self._automation_batch_template = prompt["automation_batch_template"]
        self._json_repair_system_prompt = prompt["json_repair_system_prompt"]
        self._json_repair_user_template = prompt["json_repair_user_template"]
        self._kb_section_template = prompt["kb_section_template"]

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        if context.generation_mode == "automation":
            context.test_cases = await self._generate_automation_cases(context)
            return context

        kb_section = (
            self._kb_section_template.format(samples=context.kb_samples.strip())
            if context.kb_samples.strip()
            else ""
        )
        user_content = self._user_template.format(
            kb_section=kb_section,
            requirement=context.requirement_text.strip(),
            module=context.module,
            generation_mode=context.generation_mode,
        )
        if context.generation_mode == "automation":
            user_content += self._automation_requirements

        response = await self._llm.generate_chat(
            messages=[
                Message(
                    role="system",
                    content=context.request.system_prompt_override or self._system_prompt,
                ),
                Message(role="user", content=user_content),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw_cases = _parse_cases(response.content.strip(), context.module)
        if not raw_cases:
            logger.warning(
                "test_case_workflow_empty_cases",
                module=context.module,
                raw=response.content[:200],
            )
            raise ValueError("LLM 未能生成有效的测试用例，请检查需求文档内容后重试。")

        context.test_cases = [
            GeneratedTestCase(
                title=case["title"],
                module=case["module"],
                precondition=case["precondition"],
                steps=case["steps"],
                expected=case["expected"],
                priority=case["priority"],
                case_type=case["type"],
                notes=case["notes"],
                data_setup=case["data_setup"],
                selectors=case["selectors"],
                automation_steps=case["automation_steps"],
                assertions=case["assertions"],
                business_name=case["business_name"],
                ui_display_name=case["ui_display_name"],
                page_route=case["page_route"],
                locator_chain=case["locator_chain"],
                anchor_text=case["anchor_text"],
                search_strategy=case["search_strategy"],
                expected_visibility=case["expected_visibility"],
                forbidden_locators=case["forbidden_locators"],
            )
            for case in raw_cases
        ]
        return context

    async def _generate_automation_cases(
        self,
        context: WorkflowContext,
    ) -> list[GeneratedTestCase]:
        """按功能点分批生成 UI 自动化用例，避免单次输出过长。"""
        features = _extract_feature_payloads(context.requirement_text)
        batches = _chunk_list(features, size=2) if features else [[]]
        kb_section = (
            self._kb_section_template.format(samples=context.kb_samples.strip())
            if context.kb_samples.strip()
            else ""
        )
        all_cases: list[dict] = []
        for batch_index, batch in enumerate(batches, start=1):
            feature_names = [
                str(item.get("name") or item.get("id") or f"批次{batch_index}")
                for item in batch
            ]
            user_content = self._automation_batch_template.format(
                kb_section=kb_section,
                requirement=_render_automation_batch_requirement(
                    context.requirement_text,
                    batch,
                ),
                module=context.module,
            )
            user_content += self._automation_requirements
            response = await self._llm.generate_chat(
                messages=[
                    Message(
                        role="system",
                        content=(
                            context.request.system_prompt_override
                            or self._system_prompt
                        ),
                    ),
                    Message(role="user", content=user_content),
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw_cases = _parse_cases(response.content.strip(), context.module)
            if not raw_cases:
                raw_cases = await self._repair_cases(
                    response.content.strip(),
                    context.module,
                )
            if not raw_cases:
                logger.warning(
                    "test_case_workflow_empty_cases",
                    module=context.module,
                    batch=batch_index,
                    features=feature_names,
                    raw=response.content[:200],
                )
                joined = "、".join(feature_names)
                raise ValueError(
                    f"LLM 未能生成有效的 UI 自动化用例，失败批次：{joined}。"
                )
            all_cases.extend(raw_cases)

        return [_case_from_dict(case) for case in all_cases]

    async def _repair_cases(self, raw: str, module: str) -> list[dict]:
        """对被截断或带多余文本的 JSON 输出做一次修复重试。"""
        if not raw.strip():
            return []
        response = await self._llm.generate_chat(
            messages=[
                Message(role="system", content=self._json_repair_system_prompt),
                Message(
                    role="user",
                    content=self._json_repair_user_template.format(raw=raw[:12000]),
                ),
            ],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        return _parse_cases(response.content.strip(), module)


def _parse_cases(raw: str, module: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    for candidate in [text, _find_first_json_array(text)]:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [_normalise(c, module) for c in data if isinstance(c, dict)]
        except json.JSONDecodeError:
            pass
    return []


def _find_first_json_array(text: str) -> str:
    match = re.search(r"\[[\s\S]*\]", text)
    return match.group() if match else ""


def _normalise(raw: dict, default_module: str) -> dict:
    return {
        "title": str(raw.get("title", "未命名用例")).strip(),
        "module": str(raw.get("module", default_module)).strip(),
        "precondition": str(raw.get("precondition", "无")).strip(),
        "steps": _format_numbered_text(raw.get("steps", "")),
        "expected": _format_numbered_text(raw.get("expected", "")),
        "priority": str(raw.get("priority", "P1")).strip(),
        "type": str(raw.get("type", "正向")).strip(),
        "notes": str(raw.get("notes", "")).strip(),
        "data_setup": stringify_extra(raw.get("data_setup", "")),
        "business_name": stringify_extra(raw.get("business_name", "")),
        "ui_display_name": stringify_extra(raw.get("ui_display_name", "")),
        "page_route": stringify_extra(raw.get("page_route", "")),
        "locator_chain": stringify_extra(raw.get("locator_chain", "")),
        "anchor_text": stringify_extra(raw.get("anchor_text", "")),
        "search_strategy": stringify_extra(raw.get("search_strategy", "")),
        "expected_visibility": normalise_expected_visibility(
            raw.get("expected_visibility", "")
        ),
        "forbidden_locators": stringify_extra(raw.get("forbidden_locators", "")),
        "selectors": stringify_extra(raw.get("selectors", "")),
        "automation_steps": _format_jsonish_text(raw.get("automation_steps", "")),
        "assertions": _format_jsonish_text(raw.get("assertions", "")),
    }


def _case_from_dict(case: dict) -> GeneratedTestCase:
    return GeneratedTestCase(
        title=case["title"],
        module=case["module"],
        precondition=case["precondition"],
        steps=case["steps"],
        expected=case["expected"],
        priority=case["priority"],
        case_type=case["type"],
        notes=case["notes"],
        data_setup=case["data_setup"],
        selectors=case["selectors"],
        automation_steps=case["automation_steps"],
        assertions=case["assertions"],
        business_name=case["business_name"],
        ui_display_name=case["ui_display_name"],
        page_route=case["page_route"],
        locator_chain=case["locator_chain"],
        anchor_text=case["anchor_text"],
        search_strategy=case["search_strategy"],
        expected_visibility=case["expected_visibility"],
        forbidden_locators=case["forbidden_locators"],
    )


def _format_numbered_text(value) -> str:
    parsed = _coerce_sequence(value)
    if isinstance(parsed, list):
        items = [_stringify_plain(item) for item in parsed if _stringify_plain(item)]
        return "\n".join(f"{idx}、{item}" for idx, item in enumerate(items, start=1))
    if isinstance(parsed, dict):
        items = [f"{key}：{_stringify_plain(val)}" for key, val in parsed.items()]
        return "\n".join(f"{idx}、{item}" for idx, item in enumerate(items, start=1))
    return _stringify_plain(parsed)


def _format_jsonish_text(value) -> str:
    parsed = _coerce_sequence(value)
    if isinstance(parsed, (list, dict)):
        return json.dumps(parsed, ensure_ascii=False)
    return _stringify_plain(parsed)


def _coerce_sequence(value):
    if isinstance(value, (list, dict)):
        return value
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
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def _stringify_plain(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_feature_payloads(requirement_text: str) -> list[dict]:
    marker = "确认版需求分析 JSON："
    text = requirement_text.split(marker, 1)[-1].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    features = payload.get("features", []) if isinstance(payload, dict) else []
    return [item for item in features if isinstance(item, dict)]


def _render_automation_batch_requirement(
    requirement_text: str,
    batch_features: list[dict],
) -> str:
    """只保留当前批次功能点和全局策略，降低自动化生成上下文体积。"""
    marker = "确认版需求分析 JSON："
    text = requirement_text.split(marker, 1)[-1].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return requirement_text.strip()
    if not isinstance(payload, dict):
        return requirement_text.strip()
    batch_payload = {
        "summary": payload.get("summary", ""),
        "features": batch_features,
        "state_transitions": payload.get("state_transitions", []),
        "test_strategy": payload.get("test_strategy", {}),
    }
    return "确认版需求分析 JSON：\n" + json.dumps(
        batch_payload,
        ensure_ascii=False,
        indent=2,
    )


def _chunk_list(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index:index + size] for index in range(0, len(items), size)]
