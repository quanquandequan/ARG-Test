"""RequirementIR 构建器：统一需求解析入口。"""

from __future__ import annotations

from datetime import datetime

from src.llm.base import BaseLLM
from src.llm.types import Message
from src.services.requirement_ir import RequirementIR

_SYSTEM_PROMPT = """\
你是一名资深软件测试工程师，专注于需求结构化解析。
你的任务是将需求文档转换为 RequirementIR（需求中间表示），以 JSON 格式输出。

## 信息优先级（严格遵守）
- 当前输入的需求文档是唯一的需求事实来源。
- features、acceptance_criteria、business_rules、state_machines、
  data_entities、out_of_scope 只能来自当前需求文档的明确描述。
- 知识库背景只允许用于补充 test_hints 中的回归风险提示，且必须以
  “回归参考：”开头。
- 不得把知识库中的历史页面、历史功能、历史登录态、分页、跳转、文案等
  写成当前需求功能或验收标准。
- 当知识库背景与需求文档冲突时，必须以需求文档为准。

## 输出格式（严格遵守）
只输出 JSON 对象，不加 Markdown 标记或任何解释。

JSON 结构如下：
{
  "module": "功能模块名",
  "summary": "一句话核心描述（30字以内）",
  "actors": [
    {"name": "角色名", "role": "角色职责描述"}
  ],
  "features": [
    {
      "id": "F001",
      "name": "功能名称",
      "description": "面向测试人员的功能描述（做什么、何时、对谁）",
      "priority": "P0",
      "acceptance_criteria": [
        "验收标准1（具体可测量）",
        "验收标准2"
      ],
      "test_hints": [
        "测试提示1（边界/异常/风险场景提示）"
      ],
      "dependencies": ["F002"]
    }
  ],
  "business_rules": [
    {
      "id": "R001",
      "description": "规则简述",
      "condition": "IF 条件（触发场景）",
      "outcome": "THEN 结果（系统行为）",
      "related_features": ["F001"]
    }
  ],
  "state_machines": [
    {
      "entity": "实体名（如：订单、账户）",
      "states": ["状态A", "状态B", "状态C"],
      "initial_state": "状态A",
      "transitions": [
        {
          "from_state": "状态A",
          "to_state": "状态B",
          "trigger": "触发事件",
          "guard": "前提条件（无则填空字符串）"
        }
      ]
    }
  ],
  "data_entities": [
    {
      "name": "实体名（如：登录请求）",
      "fields": [
        {
          "name": "字段名",
          "field_type": "string",
          "constraints": ["max_length=20", "required"],
          "required": true
        }
      ]
    }
  ],
  "out_of_scope": ["明确不在本次范围内的内容"]
}

## 解析原则
- features：每个独立用户操作/系统行为拆为一个 feature
  - priority：P0=核心主流程，P1=重要功能，P2=边缘
  - acceptance_criteria：具体可验证，避免"正常显示"等模糊描述
  - test_hints：直接给出边界值、异常场景、权限场景提示
- business_rules：系统必须强制执行的约束，IF/THEN 格式
- state_machines：找出有状态变化的实体（订单、账户、内容审核等）
- data_entities：找出所有请求/响应/表单的字段及约束
- 可结合知识库背景，在 test_hints 中标注变更点和回归风险，但不得改写需求事实
"""

_USER_TEMPLATE = """\
需求文档：
{requirement}

模块名称：{module}

{kb_section}
请输出 RequirementIR JSON。
"""

_KB_SECTION = """\
【历史知识库参考（辅助）】
以下是历史功能逻辑或测试用例，仅用于识别变更点、回归风险和回测范围。
不得作为当前需求事实来源；不得修改或补写需求文档没有描述的功能。
如需引用，请仅在 test_hints 中以“回归参考：”开头。

{context}

---
"""


class RequirementIRBuilder:
    """统一构建 RequirementIR。"""

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        from src.core.config import get_config

        cfg = get_config().get("req_parser", {})
        self._llm = llm
        self._temperature = (
            temperature if temperature is not None
            else float(cfg.get("temperature", 0.2))
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None
            else int(cfg.get("max_tokens", 8192))
        )
        self._system_prompt = system_prompt or cfg.get("system_prompt", "") or _SYSTEM_PROMPT

    async def build(
        self,
        requirement: str,
        module: str,
        kb_context: str = "",
    ) -> RequirementIR | None:
        """调用 LLM 并返回校验通过的 RequirementIR。"""
        if not requirement or not requirement.strip():
            return None

        kb_section = (
            _KB_SECTION.format(context=kb_context.strip())
            if kb_context.strip()
            else ""
        )
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(
                    kb_section=kb_section,
                    requirement=requirement.strip(),
                    module=module.strip() or "需求解析",
                ),
            ),
        ]

        response = await self._llm.generate_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        ir = RequirementIR.from_llm_json(response.content)
        if ir is None:
            return None

        return ir.model_copy(update={
            "module": module.strip() or "需求解析",
            "source_length": len(requirement),
            "has_kb_context": bool(kb_context.strip()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        })


def render_requirement_ir_markdown(ir: RequirementIR) -> str:
    """将 RequirementIR 渲染为便于人工阅读的 Markdown。"""
    lines: list[str] = [
        f"# RequirementIR 摘要：{ir.module}",
        "",
        f"> 生成时间：{ir.generated_at}",
        "",
        "## 摘要",
        "",
        ir.summary,
        "",
    ]

    if ir.actors:
        lines += ["## 参与角色", ""]
        for actor in ir.actors:
            lines.append(f"- **{actor.name}**：{actor.role}")
        lines.append("")

    if ir.features:
        lines += [
            "## 功能点",
            "",
            "| ID | 名称 | 优先级 | 验收标准数 | 测试提示数 |",
            "|---|---|---|---|---|",
        ]
        for feature in ir.features:
            lines.append(
                f"| {feature.id} | {feature.name} | {feature.priority}"
                f" | {len(feature.acceptance_criteria)} | {len(feature.test_hints)} |"
            )
        lines.append("")

    if ir.business_rules:
        lines += [
            "## 业务规则",
            "",
            "| ID | 描述 | 条件 | 结果 |",
            "|---|---|---|---|",
        ]
        for rule in ir.business_rules:
            lines.append(
                f"| {rule.id} | {rule.description} | {rule.condition} | {rule.outcome} |"
            )
        lines.append("")

    if ir.state_machines:
        lines += ["## 状态机", ""]
        for machine in ir.state_machines:
            lines.append(f"### {machine.entity}")
            lines.append("")
            lines.append(f"- 初始状态：{machine.initial_state or '未指定'}")
            lines.append(f"- 状态集合：{', '.join(machine.states)}")
            if machine.transitions:
                lines.append("- 状态转换：")
                for transition in machine.transitions:
                    guard = transition.guard or "无"
                    lines.append(
                        f"  - {transition.from_state} -> {transition.to_state}"
                        f"（触发：{transition.trigger}；前提：{guard}）"
                    )
            lines.append("")

    if ir.data_entities:
        lines += ["## 数据实体", ""]
        for entity in ir.data_entities:
            lines.append(f"### {entity.name}")
            lines.append("")
            if entity.fields:
                lines.append("| 字段 | 类型 | 必填 | 约束 |")
                lines.append("|---|---|---|---|")
                for field in entity.fields:
                    constraints = "、".join(field.constraints) or "-"
                    required = "是" if field.required else "否"
                    lines.append(
                        f"| {field.name} | {field.field_type} | {required} | {constraints} |"
                    )
                lines.append("")

    if ir.out_of_scope:
        lines += ["## 范围外内容", ""]
        for item in ir.out_of_scope:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)
