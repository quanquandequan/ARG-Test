"""Requirement parser tool — converts a requirements document to a RequirementIR.

Difference from ``analyze_requirements``:
  - analyze_requirements → human-readable Markdown report (good for Q&A)
  - requirement_parser   → structured RequirementIR JSON (drives downstream tools)

The RequirementIR is the canonical data contract between the Requirement Domain
and the Test Design / Execution domains.

Workflow:
  1. Agent calls ``knowledge_search`` to get existing feature background.
  2. Agent calls ``requirement_parser``.
  3. This tool calls the LLM to produce RequirementIR-shaped JSON.
  4. Validates with Pydantic; saves ``<module>_<ts>_ir.json`` +
     ``<module>_<ts>_ir_summary.md``.
  5. Returns ``[IR_FILE=<path>]`` marker so downstream tools can load the IR
     without re-passing the full JSON through the context window.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import Message
from src.services.requirement_ir import RequirementIR

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirement_ir"

# ── LLM prompts ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一名资深软件测试工程师，专注于需求结构化解析。
你的任务是将需求文档转换为 RequirementIR（需求中间表示），以 JSON 格式输出。

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
- 结合知识库背景，在 test_hints 中标注变更点和回归风险
"""

_USER_TEMPLATE = """\
{kb_section}需求文档：
{requirement}

模块名称：{module}

请输出 RequirementIR JSON。
"""

_KB_SECTION = """\
【知识库背景】以下是现有功能逻辑或测试用例，请用于识别变更点、回归风险，\
并在 test_hints 中体现：

{context}

---
"""


class RequirementParserTool(BaseTool):
    """Parse a requirements document into a structured RequirementIR.

    The IR is the typed data contract for downstream Test Design tools.
    Saves two files:
      ``<module>_<ts>_ir.json``         — full RequirementIR (Pydantic-validated)
      ``<module>_<ts>_ir_summary.md``   — human-readable summary

    Returns ``[IR_FILE=<path>]`` in the output so the Agent (or a human) can
    pass the path to ``requirement_reviewer`` or ``test_point_generator``.
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str | None = None,
        output_dir: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        from src.core.config import get_config

        cfg = get_config().get("req_parser", {})
        self._llm = llm
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self._temperature = (
            temperature if temperature is not None
            else float(cfg.get("temperature", 0.2))
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None
            else int(cfg.get("max_tokens", 8192))
        )
        self._system_prompt = (
            system_prompt or cfg.get("system_prompt", "") or _SYSTEM_PROMPT
        )

    @property
    def name(self) -> str:
        return "requirement_parser"

    @property
    def description(self) -> str:
        return (
            "将需求文档解析为结构化 RequirementIR（需求中间表示），\n"
            "供测试设计工具（test_point_generator 等）消费。\n"
            "包含：功能点（含验收标准/测试提示）、业务规则（IF/THEN）、\n"
            "状态机、数据实体字段定义。\n\n"
            "调用规范：\n"
            "- 叭嗒 app 功能：先 knowledge_search '功能名 测试用例'\n"
            "- 插件/小程序：先 knowledge_search '功能名 xmind'\n"
            "- 将搜索结果传入 kb_context\n"
            "- 输出 [IR_FILE=路径] 供后续工具使用"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "requirement": {
                    "type": "string",
                    "description": "需求文档的完整文本",
                },
                "kb_context": {
                    "type": "string",
                    "description": "来自 knowledge_search 的背景信息（推荐提供）",
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名称（用于文件命名）",
                },
                "output_dir": {
                    "type": "string",
                    "description": f"输出目录（默认 {_DEFAULT_OUTPUT_DIR}）",
                },
            },
            "required": ["requirement"],
        }

    async def execute(
        self,
        requirement: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> str:
        if not requirement or not requirement.strip():
            return "错误：请提供需求文档内容。"

        module = module.strip() or "需求解析"
        out_dir = (
            Path(output_dir.strip()) if output_dir.strip()
            else self._default_output_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt
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
                    module=module,
                ),
            ),
        ]

        # Call LLM
        response = await self._llm.generate_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Validate against RequirementIR schema
        ir = RequirementIR.from_llm_json(response.content)
        if ir is None:
            logger.warning(
                "requirement_parser_parse_failed",
                module=module,
                raw=response.content[:200],
            )
            return (
                "LLM 未能生成有效的 RequirementIR JSON，"
                "请检查需求文档格式后重试。"
            )

        # Inject metadata
        ir = ir.model_copy(update={
            "module": module,
            "source_length": len(requirement),
            "has_kb_context": bool(kb_context.strip()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        })

        # Save files
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[\\/:*?"<>|]', "_", module)
        ir_path = out_dir / f"{safe}_{ts}_ir.json"
        md_path = out_dir / f"{safe}_{ts}_ir_summary.md"

        ir_path.write_text(
            ir.model_dump_json(indent=2), encoding="utf-8"
        )
        md_path.write_text(
            _render_markdown(ir), encoding="utf-8"
        )

        logger.info(
            "requirement_parser_done",
            module=module,
            features=ir.feature_count(),
            rules=len(ir.business_rules),
        )

        lines = [
            f"需求解析完成：{module}",
            "",
            f"IR 文件：{ir_path.resolve()}",
            f"摘要报告：{md_path.resolve()}",
            "",
            ir.to_compact_summary(),
            "",
            "功能点列表：",
        ]
        for f in ir.features:
            lines.append(f"  [{f.id}] {f.name} ({f.priority})")
        if ir.business_rules:
            lines += ["", "业务规则："]
            for r in ir.business_rules:
                lines.append(f"  [{r.id}] {r.description}")
        if ir.state_machines:
            lines += ["", "状态机："]
            for sm in ir.state_machines:
                lines.append(f"  {sm.entity}：{'→'.join(sm.states[:4])}")

        lines += ["", f"[IR_FILE={ir_path.resolve()}]"]
        return "\n".join(lines)


# ── Markdown renderer ─────────────────────────────────────────────────────────

def _render_markdown(ir: RequirementIR) -> str:
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
        for a in ir.actors:
            lines.append(f"- **{a.name}**：{a.role}")
        lines.append("")

    if ir.features:
        lines += [
            "## 功能点",
            "",
            "| ID | 名称 | 优先级 | 验收标准数 | 测试提示数 |",
            "|---|---|---|---|---|",
        ]
        for f in ir.features:
            lines.append(
                f"| {f.id} | {f.name} | {f.priority}"
                f" | {len(f.acceptance_criteria)} | {len(f.test_hints)} |"
            )
        lines.append("")

        lines += ["## 功能详情", ""]
        for f in ir.features:
            lines += [f"### {f.id}：{f.name}", ""]
            if f.description:
                lines += [f"**描述**：{f.description}", ""]
            if f.acceptance_criteria:
                lines += ["**验收标准**：", ""]
                lines += [f"- {c}" for c in f.acceptance_criteria]
                lines.append("")
            if f.test_hints:
                lines += ["**测试提示**：", ""]
                lines += [f"- {h}" for h in f.test_hints]
                lines.append("")
            if f.dependencies:
                lines += [f"**依赖**：{', '.join(f.dependencies)}", ""]

    if ir.business_rules:
        lines += ["## 业务规则", ""]
        for r in ir.business_rules:
            lines += [
                f"### {r.id}：{r.description}",
                "",
                f"- **条件**：{r.condition}",
                f"- **结果**：{r.outcome}",
                "",
            ]

    for sm in ir.state_machines:
        lines += [
            f"## 状态机：{sm.entity}",
            "",
            f"**状态**：{'、'.join(sm.states)}",
            f"**初始状态**：{sm.initial_state}",
            "",
            "| 当前状态 | 触发事件 | 前提 | 目标状态 |",
            "|---|---|---|---|",
        ]
        for t in sm.transitions:
            lines.append(
                f"| {t.from_state} | {t.trigger} | {t.guard} | {t.to_state} |"
            )
        lines.append("")

    if ir.data_entities:
        lines += ["## 数据实体", ""]
        for ent in ir.data_entities:
            lines += [f"### {ent.name}", ""]
            if ent.fields:
                lines += [
                    "| 字段 | 类型 | 必填 | 约束 |",
                    "|---|---|---|---|",
                ]
                for field in ent.fields:
                    constraints = "、".join(field.constraints)
                    lines.append(
                        f"| {field.name} | {field.field_type}"
                        f" | {'是' if field.required else '否'} | {constraints} |"
                    )
            lines.append("")

    if ir.out_of_scope:
        lines += ["## 本次范围外", ""]
        lines += [f"- {item}" for item in ir.out_of_scope]
        lines.append("")

    return "\n".join(lines)
