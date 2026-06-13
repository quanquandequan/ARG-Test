"""RequirementIR 构建器：统一需求解析入口。"""

from __future__ import annotations

from datetime import datetime

from src.core.prompt_loader import require_prompt_fields
from src.llm.base import BaseLLM
from src.llm.types import Message
from src.domain.requirement.requirement_ir import RequirementIR


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
        prompt = require_prompt_fields(
            "requirement_ir_builder",
            ["system_prompt", "user_template", "kb_section_template"],
        )
        self._system_prompt = system_prompt or prompt["system_prompt"]
        self._user_template = prompt["user_template"]
        self._kb_section_template = prompt["kb_section_template"]

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
            self._kb_section_template.format(context=kb_context.strip())
            if kb_context.strip()
            else ""
        )
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(
                role="user",
                content=self._user_template.format(
                    kb_section=kb_section,
                    requirement=requirement.strip(),
                    module=module.strip() or "需求解析",
                ),
            ),
        ]

        # DeepSeek 偶发输出格式错误，最多重试 2 次
        ir = None
        for attempt in range(3):
            response = await self._llm.generate_chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            ir = RequirementIR.from_llm_json(response.content)
            if ir is not None:
                break
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
