"""Requirement reviewer tool — quality gate for requirements before test design.

Takes a RequirementIR (via file path produced by ``requirement_parser``) and
audits it for:
  - Ambiguities (vague/unmeasurable statements)
  - Gaps (missing information test engineers need)
  - Risks (technical or business risk areas)

Outputs a ReviewResult JSON + saves a Markdown review report.

Typical workflow:
    knowledge_search → requirement_parser → requirement_reviewer
                                                    │
                                            Reviewer blocks or
                                          passes to test_point_gen
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import Message
from src.services.requirement_ir import RequirementIR, ReviewResult

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirement_ir"

# ── LLM prompts ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一名资深测试架构师，专注于需求质量评审。
你的任务是对需求进行测试可行性审查，识别歧义、缺口和风险，以 JSON 格式输出审查结果。

## 输出格式（严格遵守）
只输出 JSON 对象，不加 Markdown 标记或解释文字。

{
  "overall_quality": "good | needs_clarification | poor",
  "score": 85,
  "ambiguities": [
    {
      "id": "A001",
      "location": "F001",
      "description": "歧义描述（具体指出模糊之处）",
      "suggestion": "建议澄清方式"
    }
  ],
  "gaps": [
    {
      "id": "G001",
      "description": "缺口描述（缺少什么信息）",
      "impact": "对测试的影响",
      "question": "需要向产品/开发确认的具体问题？"
    }
  ],
  "risks": [
    {
      "area": "风险区域",
      "level": "high | medium | low",
      "description": "风险描述",
      "mitigation": "测试应对策略"
    }
  ],
  "suggestions": [
    "改进建议1",
    "改进建议2"
  ]
}

## 评分标准（overall_quality & score）
- good（80-100）：验收标准清晰可测、无明显缺口、风险已识别并有应对
- needs_clarification（50-79）：有歧义或缺口但可通过澄清解决
- poor（0-49）：关键信息缺失、无法开始测试设计

## 审查重点
1. 验收标准是否具体可验证（避免"正常显示"、"响应快"等）
2. 是否覆盖了错误场景和边界条件
3. 权限和角色是否明确
4. 状态转换是否完整（是否有孤立状态）
5. 数据字段约束是否完整（类型、范围、必填性）
6. 是否有隐含的技术依赖未说明
"""

_USER_TEMPLATE_WITH_IR = """\
以下是已解析的 RequirementIR（JSON 格式），请对其进行质量审查：

{ir_json}

模块：{module}

请输出 ReviewResult JSON。
"""

_USER_TEMPLATE_RAW = """\
以下是需求文档原文，请从测试视角进行质量审查：

{requirement}

模块：{module}

请输出 ReviewResult JSON。
"""


class RequirementReviewerTool(BaseTool):
    """Audit a RequirementIR (or raw requirement text) for test-design readiness.

    Preferred input: ``ir_file`` (path from ``requirement_parser`` output).
    Fallback input: ``requirement`` (raw text) when no parser output is available.

    Saves:
      ``<module>_<ts>_review.json``  — ReviewResult (Pydantic-validated)
      ``<module>_<ts>_review.md``    — human-readable review report
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

        cfg = get_config().get("req_reviewer", {})
        self._llm = llm
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self._temperature = (
            temperature if temperature is not None
            else float(cfg.get("temperature", 0.3))
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None
            else int(cfg.get("max_tokens", 4096))
        )
        self._system_prompt = (
            system_prompt or cfg.get("system_prompt", "") or _SYSTEM_PROMPT
        )

    @property
    def name(self) -> str:
        return "requirement_reviewer"

    @property
    def description(self) -> str:
        return (
            "对需求（RequirementIR 或原文）进行质量评审，\n"
            "识别歧义（ambiguities）、信息缺口（gaps）、测试风险（risks）。\n"
            "输出质量评分（0-100）和详细的审查报告。\n\n"
            "典型用法：\n"
            "  1. requirement_parser 输出 [IR_FILE=路径]\n"
            "  2. requirement_reviewer(ir_file=路径)\n"
            "  → 质量评分 < 70 时，建议先澄清问题再进行测试设计"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "ir_file": {
                    "type": "string",
                    "description": (
                        "requirement_parser 输出的 IR JSON 文件路径（推荐）。"
                        "可从上一步工具输出的 [IR_FILE=...] 中提取。"
                    ),
                },
                "requirement": {
                    "type": "string",
                    "description": "需求原文（当 ir_file 不可用时作为替代）",
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名（用于文件命名）",
                },
                "output_dir": {
                    "type": "string",
                    "description": f"输出目录（默认 {_DEFAULT_OUTPUT_DIR}）",
                },
            },
        }

    async def execute(
        self,
        ir_file: str = "",
        requirement: str = "",
        module: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> str:
        if not ir_file.strip() and not requirement.strip():
            return "错误：请提供 ir_file（IR 文件路径）或 requirement（需求原文）之一。"

        out_dir = (
            Path(output_dir.strip()) if output_dir.strip()
            else self._default_output_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load input
        ir: RequirementIR | None = None
        if ir_file.strip():
            ir, module, user_content = self._load_from_ir_file(
                ir_file.strip(), module
            )
        else:
            module = module.strip() or "需求评审"
            user_content = _USER_TEMPLATE_RAW.format(
                requirement=requirement.strip(), module=module
            )

        # Call LLM
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]
        response = await self._llm.generate_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Parse ReviewResult
        review = ReviewResult.from_llm_json(response.content)
        if review is None:
            logger.warning(
                "requirement_reviewer_parse_failed",
                module=module,
                raw=response.content[:200],
            )
            return "LLM 未能生成有效的 ReviewResult，请重试。"

        # Save files
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[\\/:*?"<>|]', "_", module)
        review_json_path = out_dir / f"{safe}_{ts}_review.json"
        review_md_path = out_dir / f"{safe}_{ts}_review.md"

        review_json_path.write_text(
            review.model_dump_json(indent=2), encoding="utf-8"
        )
        review_md_path.write_text(
            _render_review_markdown(review, module, ir),
            encoding="utf-8",
        )

        logger.info(
            "requirement_reviewer_done",
            module=module,
            score=review.score,
            quality=review.overall_quality,
            ambiguities=len(review.ambiguities),
            gaps=len(review.gaps),
        )

        lines = [
            f"需求评审完成：{module}",
            "",
            f"评审报告：{review_md_path.resolve()}",
            "",
            review.to_compact_summary(),
            "",
        ]

        if review.ambiguities:
            lines += ["歧义问题："]
            for a in review.ambiguities[:5]:
                lines.append(f"  [{a.id}@{a.location}] {a.description}")
            if len(review.ambiguities) > 5:
                lines.append(f"  ...共 {len(review.ambiguities)} 条，详见报告")
            lines.append("")

        if review.gaps:
            lines += ["信息缺口（需向产品确认）："]
            for g in review.gaps[:3]:
                lines.append(f"  [{g.id}] {g.question}")
            lines.append("")

        high_risks = [r for r in review.risks if r.level == "high"]
        if high_risks:
            lines += ["高风险区域："]
            for r in high_risks:
                lines.append(f"  [{r.area}] {r.description}")
            lines.append("")

        if review.suggestions:
            lines += ["改进建议："]
            for s in review.suggestions[:3]:
                lines.append(f"  - {s}")
            lines.append("")

        # Quality gate hint
        if review.score < 70:
            lines.append(
                "⚠️ 质量评分低于 70，建议先澄清歧义和填补缺口，再进行测试设计。"
            )
        elif review.score >= 85:
            lines.append("✅ 需求质量良好，可以开始测试点生成（test_point_generator）。")
        else:
            lines.append("⚠️ 建议澄清部分问题后再进行测试设计。")

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_from_ir_file(
        self, ir_file: str, module: str
    ) -> tuple[RequirementIR | None, str, str]:
        """Load IR from file and build user message. Returns (ir, module, content)."""
        path = Path(ir_file)
        if not path.exists():
            raise FileNotFoundError(f"IR 文件不存在：{ir_file}")

        raw = path.read_text(encoding="utf-8")
        try:
            ir = RequirementIR.model_validate_json(raw)
            resolved_module = module.strip() or ir.module
            # Compact IR for LLM (skip metadata fields)
            ir_for_llm = ir.model_dump(
                exclude={"version", "generated_at", "source_length", "has_kb_context"}
            )
            content = _USER_TEMPLATE_WITH_IR.format(
                ir_json=json.dumps(ir_for_llm, ensure_ascii=False, indent=2),
                module=resolved_module,
            )
            return ir, resolved_module, content
        except Exception:
            # Fallback: treat as raw text
            resolved_module = module.strip() or "需求评审"
            content = _USER_TEMPLATE_RAW.format(
                requirement=raw, module=resolved_module
            )
            return None, resolved_module, content


# ── Markdown renderer ─────────────────────────────────────────────────────────

def _render_review_markdown(
    review: ReviewResult,
    module: str,
    ir: RequirementIR | None,
) -> str:
    quality_label = {
        "good": "✅ 良好",
        "needs_clarification": "⚠️ 需澄清",
        "poor": "❌ 较差",
    }
    label = quality_label.get(review.overall_quality, review.overall_quality)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        f"# 需求评审报告：{module}",
        "",
        f"> 生成时间：{ts}",
        "",
        "## 评审结论",
        "",
        f"**质量评级**：{label}",
        f"**评分**：{review.score} / 100",
        "",
    ]

    if ir:
        lines += [
            "**基于 RequirementIR**：",
            f"- 功能点：{len(ir.features)} 个",
            f"- 业务规则：{len(ir.business_rules)} 条",
            "",
        ]

    if review.ambiguities:
        lines += [
            "## 歧义问题",
            "",
            "| ID | 位置 | 描述 | 建议 |",
            "|---|---|---|---|",
        ]
        for a in review.ambiguities:
            lines.append(
                f"| {a.id} | {a.location} | {a.description} | {a.suggestion} |"
            )
        lines.append("")

    if review.gaps:
        lines += [
            "## 信息缺口",
            "",
            "| ID | 描述 | 影响 | 待确认问题 |",
            "|---|---|---|---|",
        ]
        for g in review.gaps:
            lines.append(
                f"| {g.id} | {g.description} | {g.impact} | {g.question} |"
            )
        lines.append("")

    if review.risks:
        risk_label = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低"}
        lines += [
            "## 风险评估",
            "",
            "| 区域 | 等级 | 描述 | 应对策略 |",
            "|---|---|---|---|",
        ]
        for r in review.risks:
            lvl = risk_label.get(r.level, r.level)
            lines.append(
                f"| {r.area} | {lvl} | {r.description} | {r.mitigation} |"
            )
        lines.append("")

    if review.suggestions:
        lines += ["## 改进建议", ""]
        for s in review.suggestions:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)
