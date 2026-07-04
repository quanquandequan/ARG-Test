"""高阶自动化执行工具。"""

from __future__ import annotations

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.core.logging import get_logger
from src.domain.execution import ScenarioBatchExecutionResult, ScenarioExecutionRequest
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
            "执行已生成的自动化用例 JSON，并输出执行报告。不生成新用例。"
            "默认只执行 case_id/case_title 指定的一条（都不填时默认第一条）。"
            "需要一次跑多条时用批量参数：max_cases（按 JSON 声明顺序只跑前 N 条）、"
            "exclude_types（跳过指定 type 的用例，如 ['回归测试']）、"
            "case_ids（显式指定要跑哪些用例 id）。批量参数要在同一次调用里一起传，"
            "不要多次调用本工具来跑多条用例。"
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
                    "description": "要执行的单条用例 ID（可选，与批量参数互斥）",
                },
                "case_title": {
                    "type": "string",
                    "description": "要执行的单条用例标题（可选，与批量参数互斥）",
                },
                "case_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "批量执行：显式指定要执行的用例 id 列表（可选）",
                },
                "max_cases": {
                    "type": "integer",
                    "description": (
                        "批量执行：按 JSON 中用例声明顺序只执行前 N 条"
                        "（先应用 exclude_types 过滤，再取前 N 条，可选）"
                    ),
                },
                "exclude_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "批量执行：跳过指定 type 的用例，如 ['回归测试']（可选）",
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
        case_ids: list[str] | None = None,
        max_cases: int | None = None,
        exclude_types: list[str] | None = None,
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            automation_json_path=automation_json_path,
            case_id=case_id,
            case_title=case_title,
            app_package=app_package,
            output_dir=output_dir,
            case_ids=case_ids,
            max_cases=max_cases,
            exclude_types=exclude_types,
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
        case_ids: list[str] | None = None,
        max_cases: int | None = None,
        exclude_types: list[str] | None = None,
        **kwargs,
    ) -> ToolExecutionResult:
        if not automation_json_path.strip():
            return ToolExecutionResult(content="错误：请提供 automation_json_path。")

        case_ids = case_ids or []
        exclude_types = exclude_types or []
        # 只要出现任一批量参数就走批量执行；未指定任何批量参数时沿用历史的
        # 单条执行行为（case_id/case_title 都不填则默认执行 cases[0]）。
        is_batch = bool(case_ids) or bool(exclude_types) or max_cases is not None

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

            request = ScenarioExecutionRequest(
                automation_json_path=automation_json_path,
                case_id=case_id,
                case_title=case_title,
                app_package=app_package,
                output_dir=output_dir,
                request_id=request_id,
                case_ids=case_ids,
                max_cases=max_cases,
                exclude_types=exclude_types,
            )
            if is_batch:
                batch_result = await self._workflow.execute_batch(request)
                return _format_batch_result(batch_result, request_id, output_dir)

            result = await self._workflow.execute(request)
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


def _format_batch_result(
    batch_result: ScenarioBatchExecutionResult,
    request_id: str,
    output_dir: str,
) -> ToolExecutionResult:
    """把批量执行结果整理成给用户看的摘要文本，每条用例失败不中断后续执行。"""
    lines = [
        f"批量执行完成：共 {len(batch_result.results)} 条，"
        f"通过 {batch_result.pass_count} 条，失败 {batch_result.fail_count} 条",
        f"模块：{batch_result.module}",
    ]
    if batch_result.skipped_case_ids:
        skipped = ", ".join(batch_result.skipped_case_ids)
        lines.append(f"未执行（按 exclude_types 跳过或 case_ids 未命中）：{skipped}")

    artifacts = []
    for r in batch_result.results:
        status_mark = "✅" if r.status == "PASS" else "❌"
        line = f"{status_mark} {r.case_id} - {r.title}"
        if r.status != "PASS" and r.failure_reason:
            line += f"\n   失败原因：{r.failure_reason}"
        lines.append(line)
        if r.report_artifact is not None:
            artifacts.append(r.report_artifact)
        if r.screenshot_artifact is not None:
            artifacts.append(r.screenshot_artifact)

    return ToolExecutionResult(
        content="\n".join(lines),
        data=batch_result,
        artifacts=artifacts,
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
