"""移动端自动化执行工作流。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.tools.mobile.action_tool import ActionTool
from src.agent.tools.mobile.assertion_tool import AssertionTool
from src.agent.tools.mobile.device_tool import DeviceTool
from src.agent.tools.mobile.screen_tool import ScreenTool
from src.domain.artifacts import ArtifactKind, ArtifactRecord
from src.domain.execution import (
    ExecutionStepResult,
    ScenarioExecutionRequest,
    ScenarioExecutionResult,
)
from src.mobile.driver import AppiumDriverManager
from src.services.artifact_repository import LocalArtifactRepository
from src.services.page_cache import PageCache


class ExecutionWorkflow:
    """顺序执行自动化场景中的步骤与断言。"""

    def __init__(
        self,
        driver_manager: AppiumDriverManager,
        page_cache: PageCache,
        artifacts: LocalArtifactRepository,
        vlm=None,
    ):
        self._driver_manager = driver_manager
        self._artifacts = artifacts
        self._device_tool = DeviceTool(driver_manager=driver_manager)
        self._screen_tool = ScreenTool(
            driver_manager=driver_manager,
            page_cache=page_cache,
            vlm=vlm,
        )
        self._action_tool = ActionTool(
            driver_manager=driver_manager,
            page_cache=page_cache,
        )
        self._assertion_tool = AssertionTool(driver_manager=driver_manager)

    async def execute(
        self,
        request: ScenarioExecutionRequest,
    ) -> ScenarioExecutionResult:
        payload = json.loads(Path(request.automation_json_path).read_text(encoding="utf-8"))
        case = _select_case(payload, request.case_id, request.case_title)
        module = str(case.get("module") or payload.get("module") or "自动化执行").strip()
        case_id = str(case.get("id") or "UNKNOWN")
        title = str(case.get("title") or case_id)
        step_results: list[ExecutionStepResult] = []

        if not self._driver_manager.is_connected():
            connect_result = await self._device_tool.execute(action="connect")
            success = _is_success("device_tool", "connect", connect_result)
            step_results.append(
                ExecutionStepResult(
                    stage="setup",
                    name="device_tool.connect",
                    success=success,
                    detail=connect_result,
                )
            )
            if not success:
                return await self._finalize(
                    module=module,
                    case_id=case_id,
                    title=title,
                    status="FAIL",
                    steps=step_results,
                    request=request,
                    failure_reason=connect_result,
                )

        automation_steps = list(case.get("automation_steps") or [])
        if not automation_steps:
            return await self._finalize(
                module=module,
                case_id=case_id,
                title=title,
                status="FAIL",
                steps=step_results,
                request=request,
                failure_reason="自动化用例缺少 automation_steps，无法执行。",
            )

        for idx, step in enumerate(automation_steps, start=1):
            tool_name = str(step.get("tool", "")).strip()
            action_name = str(step.get("action", "")).strip()
            result = await self._execute_step(step, request.app_package)
            success = _is_success(tool_name, action_name, result)
            step_results.append(
                ExecutionStepResult(
                    stage="step",
                    name=f"{tool_name}.{action_name or 'execute'}#{idx}",
                    success=success,
                    detail=result,
                )
            )
            if not success:
                return await self._finalize(
                    module=module,
                    case_id=case_id,
                    title=title,
                    status="FAIL",
                    steps=step_results,
                    request=request,
                    failure_reason=result,
                )

        assertions = list(case.get("assertions") or [])
        for idx, assertion in enumerate(assertions, start=1):
            result = await self._assertion_tool.execute(**assertion)
            success = result.startswith("✅ PASS")
            step_results.append(
                ExecutionStepResult(
                    stage="assertion",
                    name=f"{assertion.get('action', 'assert')}#{idx}",
                    success=success,
                    detail=result,
                )
            )
            if not success:
                return await self._finalize(
                    module=module,
                    case_id=case_id,
                    title=title,
                    status="FAIL",
                    steps=step_results,
                    request=request,
                    failure_reason=result,
                )

        return await self._finalize(
            module=module,
            case_id=case_id,
            title=title,
            status="PASS",
            steps=step_results,
            request=request,
        )

    async def _execute_step(self, step: dict[str, Any], app_package: str) -> str:
        tool_name = str(step.get("tool", "")).strip()
        params = {k: v for k, v in step.items() if k != "tool"}
        if tool_name == "device_tool":
            if (
                params.get("action") == "launch_app"
                and not params.get("app_package")
                and app_package
            ):
                params["app_package"] = app_package
            return await self._device_tool.execute(**params)
        if tool_name == "screen_tool":
            return await self._screen_tool.execute(**params)
        if tool_name == "action_tool":
            return await self._action_tool.execute(**params)
        if tool_name == "assertion_tool":
            return await self._assertion_tool.execute(**params)
        return f"错误：不支持的执行工具 {tool_name!r}。"

    async def _finalize(
        self,
        module: str,
        case_id: str,
        title: str,
        status: str,
        steps: list[ExecutionStepResult],
        request: ScenarioExecutionRequest,
        failure_reason: str = "",
    ) -> ScenarioExecutionResult:
        screenshot_artifact = await self._capture_screenshot(
            module=module,
            case_id=case_id,
            request=request,
            status=status,
        )
        report_payload = {
            "case_id": case_id,
            "title": title,
            "module": module,
            "status": status,
            "failure_reason": failure_reason,
            "screenshot_path": str(screenshot_artifact.path) if screenshot_artifact else "",
            "steps": [
                {
                    "stage": step.stage,
                    "name": step.name,
                    "success": step.success,
                    "detail": step.detail,
                }
                for step in steps
            ],
        }
        report_artifact = self._artifacts.save_json(
            ArtifactKind.EXECUTION_REPORT_JSON,
            module,
            report_payload,
            metadata=_build_metadata(request.request_id),
            suffix=f"{case_id}_{status.lower()}",
            directory=request.output_dir.strip() or None,
        )
        return ScenarioExecutionResult(
            case_id=case_id,
            title=title,
            module=module,
            status=status,
            steps=steps,
            report_artifact=report_artifact,
            screenshot_artifact=screenshot_artifact,
            failure_reason=failure_reason,
        )

    async def _capture_screenshot(
        self,
        module: str,
        case_id: str,
        request: ScenarioExecutionRequest,
        status: str,
    ) -> ArtifactRecord | None:
        if not self._driver_manager.is_connected():
            return None
        artifact = self._artifacts.allocate(
            ArtifactKind.EXECUTION_SCREENSHOT_PNG,
            module,
            ".png",
            metadata=_build_metadata(request.request_id),
            suffix=f"{case_id}_{status.lower()}",
            directory=request.output_dir.strip() or None,
        )
        try:
            saved_path = await self._driver_manager.save_screenshot(artifact.path)
        except Exception:
            return None
        return ArtifactRecord(
            artifact_id=artifact.artifact_id,
            kind=artifact.kind,
            path=saved_path,
            media_type=artifact.media_type,
            created_at=artifact.created_at,
            metadata=artifact.metadata,
        )


def _select_case(payload: dict[str, Any], case_id: str, case_title: str) -> dict[str, Any]:
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError("自动化 JSON 中未找到 cases。")

    if case_id.strip():
        for case in cases:
            if str(case.get("id", "")).strip() == case_id.strip():
                return case
        raise ValueError(f"未找到 case_id={case_id!r} 对应的自动化用例。")

    if case_title.strip():
        for case in cases:
            title = str(case.get("title", "")).strip()
            if title == case_title.strip() or case_title.strip() in title:
                return case
        raise ValueError(f"未找到 case_title={case_title!r} 对应的自动化用例。")

    return cases[0]


def _is_success(tool_name: str, action_name: str, result: str) -> bool:
    if tool_name == "assertion_tool":
        return result.startswith("✅ PASS")
    failure_markers = ("错误", "失败", "未找到", "失效", "不可用")
    return not any(marker in result for marker in failure_markers)


def _build_metadata(request_id: str) -> dict:
    metadata: dict[str, str] = {}
    if request_id:
        metadata["request_id"] = request_id
    return metadata
