"""高阶测试用例设计工具。"""

from __future__ import annotations

from pathlib import Path

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.workflows.testcase_design import TestCaseGenerationWorkflow
from src.core.logging import get_logger

logger = get_logger(__name__)


class DesignTestCasesTool(BaseTool):
    """对 Agent 暴露单一的测试用例设计入口。"""

    final_answer_mode = FINAL_ANSWER_PASSTHROUGH

    def __init__(self, workflow: TestCaseGenerationWorkflow, output_dir=None, system_prompt=None):
        from src.core.config import get_config
        cfg = get_config().get("test_generator", {})
        self._workflow = workflow
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", "./outputs/test_cases")
        )
        self._system_prompt = system_prompt or cfg.get("system_prompt", "") or ""

    @property
    def name(self) -> str:
        return "design_test_cases"

    @property
    def description(self) -> str:
        return (
            "根据已确认的需求分析 JSON 设计完整测试用例，并输出 Excel 文件。"
            "必须先完成需求确认，传入 analysis_json_path。"
            "支持 manual（人工测试用例）与 automation（移动端自动化用例）两种模式。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "analysis_json_path": {
                    "type": "string",
                    "description": (
                        "确认版需求分析 JSON 路径，"
                        "_meta.analysis_status 必须为 confirmed"
                    ),
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名称（可选）",
                },
                "output_dir": {
                    "type": "string",
                    "description": f"输出目录，默认为 {self._default_output_dir}",
                },
                "generation_mode": {
                    "type": "string",
                    "enum": ["manual", "automation"],
                    "description": "生成模式：manual=人工测试用例，automation=自动化用例",
                },
            },
            "required": ["analysis_json_path"],
        }

    async def execute(
        self,
        analysis_json_path: str = "",
        module: str = "",
        output_dir: str = "",
        generation_mode: str = "manual",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            analysis_json_path=analysis_json_path,
            module=module,
            output_dir=output_dir,
            generation_mode=generation_mode,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        analysis_json_path: str = "",
        module: str = "",
        output_dir: str = "",
        generation_mode: str = "manual",
        request_id: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        if not analysis_json_path or not analysis_json_path.strip():
            return ToolExecutionResult(
                content=(
                    "错误：请先完成需求确认并生成确认版需求分析 JSON，"
                    "然后提供 analysis_json_path。"
                )
            )

        try:
            result = await self._workflow.run_from_analysis_json(
                analysis_json_path=analysis_json_path,
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
                "design_test_cases_failed",
                module=module,
                request_id=request_id,
            )
            raise
        except Exception:
            logger.exception(
                "design_test_cases_failed",
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
                "analysis_json_path": analysis_json_path,
                "output_dir": output_dir.strip() or str(self._default_output_dir),
            },
        )
