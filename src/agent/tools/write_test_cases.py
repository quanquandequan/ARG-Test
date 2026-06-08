"""测试用例生成工具：对接测试设计工作流。"""

from __future__ import annotations

from pathlib import Path

from src.agent.base_tool import BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.application.requirement_services import TestCaseGenerationService
from src.core.logging import get_logger
from src.domain.requirements import TestCaseGenerationData

logger = get_logger(__name__)


class WriteTestCasesTool(BaseTool):
    """面向 Agent 的测试用例生成服务适配器。"""

    __test__ = False

    def __init__(
        self,
        service: TestCaseGenerationService,
        output_dir: str | None = None,
        system_prompt: str | None = None,
    ):
        from src.core.config import get_config

        cfg = get_config().get("test_generator", {})
        self._service = service
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", "./outputs/test_cases")
        )
        self._system_prompt = system_prompt or cfg.get("system_prompt", "") or ""

    @property
    def name(self) -> str:
        return "write_test_cases"

    @property
    def description(self) -> str:
        return (
            "根据需求文档和知识库现有用例格式，自动生成完整测试用例并保存为 Excel 文件。"
            "调用前请先使用 knowledge_search 获取现有测试用例样本以对齐格式和风格。"
            "返回生成的 Excel 文件路径和用例数量。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "requirement": {
                    "type": "string",
                    "description": "需求文档的完整文本内容",
                },
                "kb_samples": {
                    "type": "string",
                    "description": "从 knowledge_search 获取的现有测试用例样本（格式参考，可选）",
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名称，用于用例分组和文件命名（可选）",
                },
                "output_dir": {
                    "type": "string",
                    "description": f"Excel 输出目录，默认为 {self._default_output_dir}",
                },
                "generation_mode": {
                    "type": "string",
                    "enum": ["manual", "automation"],
                    "description": "生成模式：manual=人工测试用例，automation=UI自动化用例",
                },
            },
            "required": ["requirement"],
        }

    async def execute(
        self,
        requirement: str = "",
        kb_samples: str = "",
        module: str = "",
        output_dir: str = "",
        generation_mode: str = "manual",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            requirement=requirement,
            kb_samples=kb_samples,
            module=module,
            output_dir=output_dir,
            generation_mode=generation_mode,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        requirement: str = "",
        kb_samples: str = "",
        module: str = "",
        output_dir: str = "",
        generation_mode: str = "manual",
        request_id: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        if not requirement or not requirement.strip():
            return ToolExecutionResult(content="错误：请提供需求文档内容。")

        try:
            result = await self._service.generate_from_text(
                requirement=requirement,
                kb_samples=kb_samples,
                module=module,
                output_dir=output_dir or str(self._default_output_dir),
                generation_mode=generation_mode,
                system_prompt_override=self._system_prompt,
                request_id=request_id,
                use_artifact_repository=True,
            )
        except ValueError as exc:
            if "LLM 未能生成" in str(exc) or "错误：" in str(exc):
                return ToolExecutionResult(content=str(exc))
            logger.exception(
                "write_test_cases_failed",
                module=module,
                request_id=request_id,
            )
            raise
        except Exception:
            logger.exception(
                "write_test_cases_failed",
                module=module,
                request_id=request_id,
            )
            raise
        artifacts = [result.workbook_artifact]
        if result.automation_json_artifact is not None:
            artifacts.append(result.automation_json_artifact)
        return ToolExecutionResult(
            content=result.summary,
            data=result.generation,
            artifacts=artifacts,
            metadata={
                "request_id": request_id,
                "output_dir": output_dir.strip() or str(self._default_output_dir),
            },
        )


def render_generation_summary(generation: TestCaseGenerationData) -> str:
    """为仅持有生成数据的旧调用方渲染摘要。"""
    positive = sum(1 for case in generation.cases if case.case_type in ("正向", "功能"))
    negative = len(generation.cases) - positive
    return "\n".join([
        "已生成测试用例 Excel 文件：",
        f"模块：{generation.module}",
        f"生成模式：{generation.generation_mode}",
        f"用例数量：{len(generation.cases)} 条",
        f"（覆盖正向 {positive} 条，反向/边界/异常 {negative} 条）",
    ])
