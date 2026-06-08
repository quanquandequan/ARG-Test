"""Generated test case view generator node."""

from __future__ import annotations

import json
import re

from src.application.exporters.common import (
    normalise_expected_visibility,
    stringify_extra,
)
from src.application.workflow_nodes.base import WorkflowContext, WorkflowNode
from src.core.logging import get_logger
from src.domain.requirements import GeneratedTestCase
from src.llm.base import BaseLLM
from src.llm.types import Message

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
你是一名资深软件测试工程师，擅长根据需求文档编写完整的测试用例。
你的任务是将需求文档转换为结构化的测试用例列表，以 JSON 数组格式输出。

## 信息优先级
- 当前输入的【需求文档内容】是唯一的业务事实来源。
- 知识库样本只用于参考 Excel 字段、用例粒度、步骤写法和术语风格。
- 当知识库样本与需求文档冲突时，必须以需求文档为准。
- 不得从知识库样本中继承需求文档没有写明的业务规则。

输出格式要求：
- 只输出 JSON 数组，不加任何 Markdown 标记或解释文字
- 每个对象字段固定为 title/module/precondition/steps/expected/priority/type/notes
- automation 模式额外输出 data_setup/business_name/ui_display_name/page_route/
  locator_chain/anchor_text/search_strategy/expected_visibility/forbidden_locators/
  selectors/automation_steps/assertions
"""

USER_TEMPLATE = """\
{kb_section}需求文档内容：
{requirement}

模块名称：{module}
生成模式：{generation_mode}

请为上述需求生成完整的测试用例，输出 JSON 数组。
"""

AUTOMATION_REQUIREMENTS = """\

automation 模式额外要求：
- 用例必须能被移动端自动化 agent 执行，不要只写人工描述。
- 必须区分需求/业务名称和真实 UI 文案。
- 禁止把不可见的需求名、组件名、Card 名当作定位词。
- page_route 必须写成可执行导航路径。
- locator_chain 必须使用组合定位。
- search_strategy 必须包含滚动方向、最大滚动次数、停止条件。
- 存在展示态/空态/隐藏态的数据规则时，必须拆成多条用例。
"""

KB_SECTION_TEMPLATE = """\
以下是知识库中现有的测试用例样本，请参考其描述风格、粒度和术语。
注意：样本不是本次需求的事实来源，不要复制样本里的业务规则或前置条件。

{samples}

---
"""


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
        self._system_prompt = system_prompt or cfg.get("system_prompt", "") or SYSTEM_PROMPT

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        kb_section = (
            KB_SECTION_TEMPLATE.format(samples=context.kb_samples.strip())
            if context.kb_samples.strip()
            else ""
        )
        user_content = USER_TEMPLATE.format(
            kb_section=kb_section,
            requirement=context.requirement_text.strip(),
            module=context.module,
            generation_mode=context.generation_mode,
        )
        if context.generation_mode == "automation":
            user_content += AUTOMATION_REQUIREMENTS

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
        "steps": str(raw.get("steps", "")).strip(),
        "expected": str(raw.get("expected", "")).strip(),
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
        "automation_steps": stringify_extra(raw.get("automation_steps", "")),
        "assertions": stringify_extra(raw.get("assertions", "")),
    }
