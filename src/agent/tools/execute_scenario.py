"""高阶自动化执行工具。"""

from __future__ import annotations

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.core.logging import get_logger
from src.domain.execution import ScenarioExecutionRequest
from src.workflows.execution import ExecutionWorkflow

logger = get_logger(__name__)


class ExecuteScenarioTool(BaseTool):
    """对 Agent 暴露单一的自动化执行入口。"""

    final_answer_mode = FINAL_ANSWER_PASSTHROUGH

    def __init__(self, workflow: ExecutionWorkflow):
        self._workflow = workflow

    @property
    def name(self) -> str:
        return "execute_scenario"

    @property
    def description(self) -> str:
        return (
            "执行已生成的自动化用例 JSON，并输出执行报告。"
            "仅接受 automation_json_path 作为执行输入；不生成新用例。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "automation_json_path": {
                    "type": "string",
                    "description": "自动化测试用例 JSON 文件路径",
                },
                "case_id": {
                    "type": "string",
                    "description": "要执行的用例 ID（可选）",
                },
                "case_title": {
                    "type": "string",
                    "description": "要执行的用例标题（可选）",
                },
                "app_package": {
                    "type": "string",
                    "description": "应用包名；当自动化步骤未显式提供时可补充",
                },
                "output_dir": {
                    "type": "string",
                    "description": "执行报告与截图输出目录（可选）",
                },
            },
            "required": ["automation_json_path"],
        }

    async def execute(
        self,
        automation_json_path: str = "",
        case_id: str = "",
        case_title: str = "",
        app_package: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            automation_json_path=automation_json_path,
            case_id=case_id,
            case_title=case_title,
            app_package=app_package,
            output_dir=output_dir,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        automation_json_path: str = "",
        case_id: str = "",
        case_title: str = "",
        app_package: str = "",
        output_dir: str = "",
        request_id: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        if not automation_json_path.strip():
            return ToolExecutionResult(content="错误：请提供 automation_json_path。")

        try:
            import json
            from pathlib import Path

            payload = json.loads(Path(automation_json_path.strip()).read_text(encoding="utf-8"))
            if _looks_like_analysis_graph_json(payload):
                return ToolExecutionResult(
                    content=(
                        "错误：当前输入是确认版需求分析 JSON，请改用 design_test_cases 生成用例；"
                        "execute_scenario 只接受自动化用例 JSON。"
                    )
                )

            result = await self._workflow.execute(
                ScenarioExecutionRequest(
                    automation_json_path=automation_json_path,
                    case_id=case_id,
                    case_title=case_title,
                    app_package=app_package,
                    output_dir=output_dir,
                    request_id=request_id,
                )
            )
        except ValueError as exc:
            return ToolExecutionResult(content=f"执行失败：{exc}")
        except Exception:
            logger.exception(
                "execute_scenario_failed",
                automation_json_path=automation_json_path,
                case_id=case_id,
                case_title=case_title,
                request_id=request_id,
            )
            raise

        lines = [
            f"执行结果：{result.status}",
            f"模块：{result.module}",
            f"用例：{result.case_id} - {result.title}",
        ]
        if result.failure_reason:
            lines.append(f"失败原因：{result.failure_reason}")
        if result.report_artifact is not None:
            lines.append(f"执行报告：{result.report_artifact.path}")
        if result.screenshot_artifact is not None:
            lines.append(f"截图：{result.screenshot_artifact.path}")

        return ToolExecutionResult(
            content="\n".join(lines),
            data=result,
            artifacts=[
                artifact
                for artifact in (
                    result.report_artifact,
                    result.screenshot_artifact,
                )
                if artifact is not None
            ],
            metadata={
                "request_id": request_id,
                "output_dir": output_dir.strip(),
            },
        )



def _looks_like_analysis_graph_json(payload: dict) -> bool:
    """识别 confirmed req_graph.json。"""
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("features"), list):
        return False
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        return False
    return str(meta.get("analysis_status", "")).strip().lower() == "confirmed"
