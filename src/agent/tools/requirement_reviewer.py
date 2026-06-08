"""需求评审工具：测试设计前的需求质量门禁。

接收 RequirementIR（通过 ``requirement_parser`` 生成的文件路径）并审查：
  - 歧义（含糊或不可度量的表述）
  - 缺口（测试工程师所需但缺失的信息）
  - 风险（技术或业务风险区域）

输出 ReviewResult JSON，并保存 Markdown 评审报告。

典型工作流：
    knowledge_search → requirement_parser → requirement_reviewer
                                                    │
                                            评审阻塞或
                                          放行至 test_point_gen
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.application.artifact_repository import LocalArtifactRepository
from src.core.logging import get_logger
from src.domain.artifacts import ArtifactKind
from src.llm.base import BaseLLM
from src.llm.types import Message
from src.services.requirement_ir import RequirementIR, ReviewResult

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirement_ir"

# ── LLM 提示词 ───────────────────────────────────────────────────────────────

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
    """审查 RequirementIR（或原始需求文本）的测试设计就绪度。

    推荐输入：``ir_file``（来自 ``requirement_parser`` 输出的路径）。
    兜底输入：无解析结果时使用 ``requirement``（原始文本）。

    保存：
      ``<module>_<ts>_review.json``  — ReviewResult（已通过 Pydantic 校验）
      ``<module>_<ts>_review.md``    — 面向人阅读的评审报告
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
        self._artifacts = LocalArtifactRepository(base_dir=str(self._default_output_dir))

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
                "ir_json": {
                    "type": "string",
                    "description": "RequirementIR JSON 字符串（内部复合工具使用）。",
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
        ir_json: str = "",
        requirement: str = "",
        module: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            ir_file=ir_file,
            ir_json=ir_json,
            requirement=requirement,
            module=module,
            output_dir=output_dir,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        ir_file: str = "",
        ir_json: str = "",
        requirement: str = "",
        module: str = "",
        output_dir: str = "",
        request_id: str = "",
        persist: bool = True,
        **kwargs,
    ) -> ToolExecutionResult:
        if not ir_file.strip() and not ir_json.strip() and not requirement.strip():
            return ToolExecutionResult(
                content=(
                    "错误：请提供 ir_file（IR 文件路径）、ir_json 或 "
                    "requirement（需求原文）之一。"
                )
            )

        out_dir = (
            Path(output_dir.strip()) if output_dir.strip()
            else self._default_output_dir
        )

        # 加载输入
        ir: RequirementIR | None = None
        if ir_json.strip():
            ir, module, user_content = self._load_from_ir_json(ir_json.strip(), module)
        elif ir_file.strip():
            ir, module, user_content = self._load_from_ir_file(
                ir_file.strip(), module
            )
        else:
            module = module.strip() or "需求评审"
            user_content = _USER_TEMPLATE_RAW.format(
                requirement=requirement.strip(), module=module
            )

        # 调用 LLM
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]
        response = await self._llm.generate_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # 解析 ReviewResult
        review = ReviewResult.from_llm_json(response.content)
        if review is None:
            logger.warning(
                "requirement_reviewer_parse_failed",
                module=module,
                raw=response.content[:200],
            )
            return ToolExecutionResult(content="LLM 未能生成有效的 ReviewResult，请重试。")

        if not persist:
            logger.info(
                "requirement_reviewer_draft_done",
                module=module,
                score=review.score,
                quality=review.overall_quality,
                ambiguities=len(review.ambiguities),
                gaps=len(review.gaps),
            )
            return ToolExecutionResult(
                content=_render_review_summary(
                    review,
                    module,
                    report_path="",
                    include_gate=True,
                    draft=True,
                ),
                data=review,
                metadata={
                    "request_id": request_id,
                    "analysis_status": "draft",
                    "persisted": False,
                },
            )

        metadata = _build_request_metadata(request_id)
        review_json_artifact = self._artifacts.save_json(
            ArtifactKind.REQUIREMENT_REVIEW_JSON,
            module,
            review.model_dump(),
            metadata=metadata,
            suffix="review",
            directory=out_dir,
        )
        review_md_artifact = self._artifacts.save_text(
            ArtifactKind.REQUIREMENT_REVIEW_MARKDOWN,
            module,
            _render_review_markdown(review, module, ir),
            extension=".md",
            metadata=metadata,
            suffix="review",
            directory=out_dir,
        )

        logger.info(
            "requirement_reviewer_done",
            module=module,
            score=review.score,
            quality=review.overall_quality,
            ambiguities=len(review.ambiguities),
            gaps=len(review.gaps),
        )

        return ToolExecutionResult(
            content=_render_review_summary(
                review,
                module,
                report_path=str(review_md_artifact.path),
                include_gate=True,
                draft=False,
            ),
            data=review,
            artifacts=[review_json_artifact, review_md_artifact],
            metadata={
                "request_id": request_id,
                "output_dir": str(out_dir.resolve()),
            },
        )

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _load_from_ir_file(
        self, ir_file: str, module: str
    ) -> tuple[RequirementIR | None, str, str]:
        """从文件加载 IR 并构建用户消息，返回 (ir, module, content)。"""
        path = Path(ir_file)
        if not path.exists():
            raise FileNotFoundError(f"IR 文件不存在：{ir_file}")

        raw = path.read_text(encoding="utf-8")
        try:
            ir = RequirementIR.model_validate_json(raw)
            resolved_module = module.strip() or ir.module
            # 为 LLM 压缩 IR（跳过元数据字段）
            ir_for_llm = ir.model_dump(
                exclude={"version", "generated_at", "source_length", "has_kb_context"}
            )
            content = _USER_TEMPLATE_WITH_IR.format(
                ir_json=json.dumps(ir_for_llm, ensure_ascii=False, indent=2),
                module=resolved_module,
            )
            return ir, resolved_module, content
        except Exception:
            # 兜底：按原始文本处理
            resolved_module = module.strip() or "需求评审"
            content = _USER_TEMPLATE_RAW.format(
                requirement=raw, module=resolved_module
            )
            return None, resolved_module, content

    def _load_from_ir_json(
        self,
        ir_json: str,
        module: str,
    ) -> tuple[RequirementIR | None, str, str]:
        """从内存中的 IR JSON 构建评审消息。"""
        try:
            ir = RequirementIR.model_validate_json(ir_json)
            resolved_module = module.strip() or ir.module
            ir_for_llm = ir.model_dump(
                exclude={"version", "generated_at", "source_length", "has_kb_context"}
            )
            content = _USER_TEMPLATE_WITH_IR.format(
                ir_json=json.dumps(ir_for_llm, ensure_ascii=False, indent=2),
                module=resolved_module,
            )
            return ir, resolved_module, content
        except Exception:
            resolved_module = module.strip() or "需求评审"
            content = _USER_TEMPLATE_RAW.format(
                requirement=ir_json,
                module=resolved_module,
            )
            return None, resolved_module, content


# ── Markdown 渲染器 ──────────────────────────────────────────────────────────

def _render_review_summary(
    review: ReviewResult,
    module: str,
    *,
    report_path: str = "",
    include_gate: bool = True,
    draft: bool = False,
) -> str:
    """渲染评审结果摘要；draft 模式不包含落盘路径。"""
    title = "需求评审草稿完成" if draft else "需求评审完成"
    lines = [
        f"{title}：{module}",
        "",
    ]
    if report_path:
        lines += [f"评审报告：{report_path}", ""]
    lines += [review.to_compact_summary(), ""]

    if review.ambiguities:
        lines += ["歧义问题："]
        for ambiguity in review.ambiguities[:5]:
            lines.append(
                f"  [{ambiguity.id}@{ambiguity.location}] {ambiguity.description}"
            )
        if len(review.ambiguities) > 5:
            lines.append(f"  ...共 {len(review.ambiguities)} 条，详见报告")
        lines.append("")

    if review.gaps:
        lines += ["信息缺口（需向产品确认）："]
        for gap in review.gaps[:3]:
            lines.append(f"  [{gap.id}] {gap.question}")
        lines.append("")

    high_risks = [risk for risk in review.risks if risk.level == "high"]
    if high_risks:
        lines += ["高风险区域："]
        for risk in high_risks:
            lines.append(f"  [{risk.area}] {risk.description}")
        lines.append("")

    if review.suggestions:
        lines += ["改进建议："]
        for suggestion in review.suggestions[:3]:
            lines.append(f"  - {suggestion}")
        lines.append("")

    if include_gate:
        if review.score < 70:
            lines.append("⚠️ 质量评分低于 70，建议先澄清歧义和填补缺口。")
        elif review.score >= 85:
            lines.append("✅ 需求质量良好，但仍需确认是否直接生成最终 JSON。")
        else:
            lines.append("⚠️ 建议澄清部分问题后再生成最终 JSON。")

    return "\n".join(lines)


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


def _build_request_metadata(request_id: str) -> dict:
    metadata: dict[str, str] = {}
    if request_id:
        metadata["request_id"] = request_id
    return metadata
