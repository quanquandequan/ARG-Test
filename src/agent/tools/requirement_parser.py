"""需求解析工具：将需求文档转换为 RequirementIR。"""

from __future__ import annotations

from pathlib import Path

from src.agent.base_tool import BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.services.artifact_repository import LocalArtifactRepository
from src.core.logging import get_logger
from src.domain.artifacts import ArtifactKind
from src.llm.base import BaseLLM
from src.services.requirement_ir import RequirementIR
from src.services.requirement_ir_builder import (
    RequirementIRBuilder,
    render_requirement_ir_markdown,
)

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirement_ir"


class RequirementParserTool(BaseTool):
    """将需求文档解析为结构化 RequirementIR。"""

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
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self._system_prompt = system_prompt or cfg.get("system_prompt", "") or None
        self._builder = RequirementIRBuilder(
            llm=llm,
            system_prompt=self._system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._artifacts = LocalArtifactRepository(base_dir=str(self._default_output_dir))

    @property
    def name(self) -> str:
        return "requirement_parser"

    @property
    def description(self) -> str:
        return (
            "将需求文档解析为结构化 RequirementIR（需求中间表示），\n"
            "供测试设计工具消费。\n"
            "包含：功能点（含验收标准/测试提示）、业务规则（IF/THEN）、\n"
            "状态机、数据实体字段定义。\n\n"
            "调用规范：\n"
            "- 当前需求文档是唯一需求事实来源\n"
            "- kb_context 仅用于历史回归风险提示，不得补写需求功能\n"
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
                    "description": "来自知识检索工具的背景信息（推荐提供）",
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
        result = await self.execute_typed(
            requirement=requirement,
            kb_context=kb_context,
            module=module,
            output_dir=output_dir,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        requirement: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        request_id: str = "",
        persist: bool = True,
        **kwargs,
    ) -> ToolExecutionResult:
        if not requirement or not requirement.strip():
            return ToolExecutionResult(content="错误：请提供需求文档内容。")

        resolved_module = module.strip() or "需求解析"
        out_dir = (
            Path(output_dir.strip()) if output_dir.strip()
            else self._default_output_dir
        )
        ir = await self._builder.build(
            requirement=requirement,
            module=resolved_module,
            kb_context=kb_context,
        )
        if ir is None:
            logger.warning("requirement_parser_parse_failed", module=resolved_module)
            return ToolExecutionResult(
                content="LLM 未能生成有效的 RequirementIR JSON，请检查需求文档格式后重试。"
            )

        if not persist:
            return ToolExecutionResult(
                content="\n".join([
                    f"需求解析草稿完成：{resolved_module}",
                    "",
                    ir.to_compact_summary(),
                ]),
                data=ir,
                metadata={
                    "request_id": request_id,
                    "analysis_status": "draft",
                    "persisted": False,
                },
            )

        metadata = _build_request_metadata(request_id)
        json_artifact = self._artifacts.save_json(
            ArtifactKind.REQUIREMENT_IR_JSON,
            resolved_module,
            ir.model_dump(),
            metadata=metadata,
            suffix="ir",
            directory=out_dir,
        )
        markdown_artifact = self._artifacts.save_text(
            ArtifactKind.REQUIREMENT_IR_MARKDOWN,
            resolved_module,
            render_requirement_ir_markdown(ir),
            extension=".md",
            metadata=metadata,
            suffix="ir_summary",
            directory=out_dir,
        )

        logger.info(
            "requirement_parser_done",
            module=resolved_module,
            features=ir.feature_count(),
            rules=len(ir.business_rules),
        )

        lines = [
            f"需求解析完成：{resolved_module}",
            "",
            f"IR 文件：{json_artifact.path}",
            f"摘要报告：{markdown_artifact.path}",
            "",
            ir.to_compact_summary(),
            "",
            "功能点列表：",
        ]
        for feature in ir.features:
            lines.append(f"  [{feature.id}] {feature.name} ({feature.priority})")
        if ir.business_rules:
            lines += ["", "业务规则："]
            for rule in ir.business_rules:
                lines.append(f"  [{rule.id}] {rule.description}")
        if ir.state_machines:
            lines += ["", "状态机："]
            for state_machine in ir.state_machines:
                lines.append(f"  {state_machine.entity}：{'→'.join(state_machine.states[:4])}")

        lines += ["", f"[IR_FILE={json_artifact.path}]"]
        return ToolExecutionResult(
            content="\n".join(lines),
            data=ir,
            artifacts=[json_artifact, markdown_artifact],
            metadata={
                "request_id": request_id,
                "output_dir": str(out_dir.resolve()),
            },
        )


def _render_markdown(ir: RequirementIR) -> str:
    """兼容旧测试导出的 Markdown 渲染函数。"""
    return render_requirement_ir_markdown(ir)


def _build_request_metadata(request_id: str) -> dict:
    metadata: dict[str, str] = {}
    if request_id:
        metadata["request_id"] = request_id
    return metadata
