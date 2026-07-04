"""移动端自动化执行工作流。"""

from __future__ import annotations

import asyncio
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
    ScenarioBatchExecutionResult,
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
        self._page_cache = page_cache
        self._assertion_tool = AssertionTool(driver_manager=driver_manager)

    async def execute(
        self,
        request: ScenarioExecutionRequest,
    ) -> ScenarioExecutionResult:
        """执行单条自动化场景（历史行为：未指定 case_id/case_title 时默认执行 cases[0]）。"""
        payload = json.loads(Path(request.automation_json_path).read_text(encoding="utf-8"))
        case = _select_case(payload, request.case_id, request.case_title)
        return await self._run_case(case, payload, request)

    async def execute_batch(
        self,
        request: ScenarioExecutionRequest,
    ) -> ScenarioBatchExecutionResult:
        """批量顺序执行多条自动化场景。

        选取规则（见 ``_select_cases``）：显式给定 ``case_ids`` 时按给定顺序精确
        匹配；否则按 JSON 声明顺序取全部用例，先按 ``exclude_types`` 过滤（如
        跳过"回归测试"），再按 ``max_cases`` 截取前 N 条。用例之间共用同一个
        设备连接顺序执行，某条失败不会中断后续用例，最终汇总成批量结果，
        由调用方（execute_scenario 工具）一次性返回给用户。

        **不要在 case 之间插入任何"无条件复位滚动"的动作**（历史上踩过这个坑）：
        `ActionTool._scroll_until_condition()` 在每次尝试前都会先检查目标是否
        已经在当前屏幕上可见，已经展示正常时不会做任何多余滑动（"共滑动 0 次"）；
        如果在这之前额外插入一段盲目的"先滑到顶再说"，反而会把原本已经展示
        正常的模块滑走，且反复顶到页面顶部很容易触发 App 自身的下拉刷新，
        导致数据/状态被意外重置，比什么都不做还糟——多滑动只会增加风险，
        不会增加可靠性。多条 case 共用同一个 App 会话本身没有问题，只要每条
        case 自己的 scroll 步骤设计正确（正确的停止条件 + 合理的 max_swipes），
        就应该信任"检查后按需滚动"这个已有机制，而不是靠额外的复位动作兜底。
        """
        payload = json.loads(Path(request.automation_json_path).read_text(encoding="utf-8"))
        module = str(payload.get("module") or "自动化执行").strip()
        selected, skipped = _select_cases(
            payload,
            case_ids=request.case_ids,
            max_cases=request.max_cases,
            exclude_types=request.exclude_types,
        )
        results = [await self._run_case(case, payload, request) for case in selected]
        return ScenarioBatchExecutionResult(
            module=module, results=results, skipped_case_ids=skipped
        )

    async def _run_case(
        self,
        case: dict[str, Any],
        payload: dict[str, Any],
        request: ScenarioExecutionRequest,
    ) -> ScenarioExecutionResult:
        """执行单条已解析出的用例字典，产出该用例的执行结果。"""
        module = str(case.get("module") or payload.get("module") or "自动化执行").strip()
        case_id = str(case.get("id") or "UNKNOWN")
        title = str(case.get("title") or case_id)
        step_results: list[ExecutionStepResult] = []

        # 用主动探活代替单纯判断本地引用是否非空：CLI 长会话里，Appium 服务端
        # 可能因 newCommandTimeout 到期等原因已销毁会话，但本地 driver 引用
        # 仍然非空，导致这里误判"已连接"而跳过重连，后续步骤直接抛
        # NoSuchDriverError。probe_session_alive 探测到会话已死时会自动清空
        # 本地引用，这里才能正确走到重新 connect 的分支。
        if not await self._driver_manager.probe_session_alive():
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

        for idx, raw_step in enumerate(automation_steps, start=1):
            step = self._normalize_step(raw_step)
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

            if tool_name == "device_tool" and action_name == "launch_app":
                # "已在前台" 说明 device_tool 跳过了 activate_app，无需等待 splash
                app_already_foreground = "已在前台" in result
                if not app_already_foreground:
                    settle_result = await self._settle_post_launch_ui()
                    if settle_result is not None:
                        settle_success = _is_success(
                            "device_tool",
                            "settle_post_launch_ui",
                            settle_result,
                        )
                        step_results.append(
                            ExecutionStepResult(
                                stage="setup",
                                name="device_tool.settle_post_launch_ui",
                                success=settle_success,
                                detail=settle_result,
                            )
                        )
                        if not settle_success:
                            return await self._finalize(
                                module=module,
                                case_id=case_id,
                                title=title,
                                status="FAIL",
                                steps=step_results,
                                request=request,
                                failure_reason=settle_result,
                            )
                    # 等待启动动画/广告页完全消失后再执行手势
                    await asyncio.sleep(5.0)

        assertions = list(case.get("assertions") or [])
        for idx, assertion in enumerate(assertions, start=1):
            try:
                result = await self._assertion_tool.execute(**assertion)
            except Exception as e:
                result = _describe_driver_crash("assertion_tool", e)
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
        """执行单个自动化步骤。

        踩过的坑：各 *_tool 内部只捕获了它们自己预期内的失败（如 swipe 参数
        错误），但 UiAutomator2 服务端在设备侧崩溃（``instrumentation process
        is not running``，常见于长时间连续操作后的偶发环境抖动）时，Appium
        会直接抛出未捕获的 ``WebDriverException``；这类异常此前会一路冒泡
        穿透 `_run_case` / `execute_batch`，把整个批量执行连同已经跑完、
        本该保留的前面几条 case 结果一起打断（`execute_scenario` 变成一次
        性失败，Agent 只能原样转述一坨 Python 堆栈）。这里统一兜底捕获，
        转换成普通的步骤失败文案，交给调用方按"这一条 case 失败"正常处理，
        不再让设备侧崩溃波及批量执行的其余用例——下一条 case 开始前的
        `probe_session_alive()` 探活会自动发现会话已死并重新连接。
        """
        tool_name = str(step.get("tool", "")).strip()
        params = {k: v for k, v in step.items() if k != "tool"}
        try:
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
        except Exception as e:
            return _describe_driver_crash(tool_name, e)

    def _normalize_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """兼容历史生成产物中的动作别名和参数命名。"""
        normalized = dict(step)
        tool_name = str(normalized.get("tool", "")).strip()
        action_name = str(normalized.get("action", "")).strip()

        if tool_name == "screen_tool" and action_name == "capture_screenshot":
            normalized["action"] = "get_screenshot"

        if tool_name == "action_tool":
            if "max_scrolls" in normalized and "max_swipes" not in normalized:
                normalized["max_swipes"] = normalized.pop("max_scrolls")
            target_type = str(normalized.get("target_type", "")).strip().lower()
            if target_type in {"classname", "class_name", "locator"}:
                normalized["target_type"] = "class"

        return normalized

    async def _settle_post_launch_ui(self) -> str | None:
        """在业务步骤开始前，尽量收敛启动页/广告页。"""
        await asyncio.sleep(2.0)  # 等待启动动画结束，1s 不够
        for attempt in range(2):
            try:
                parsed = await self._driver_manager.get_parsed_screen()
                activity = await self._driver_manager.get_current_activity()
            except Exception as e:
                return _describe_driver_crash("device_tool", e)
            if not _looks_like_splash_screen(parsed, activity):
                return None

            close_element = _find_first_visible_text(parsed, _SPLASH_CLOSE_TEXTS)
            if close_element is None:
                visible_labels = "、".join(parsed.visible_labels(limit=6)) or "无明显文案"
                return (
                    "启动应用后仍停留在启动/广告页，且未找到可安全点击的关闭入口。"
                    f"当前 Activity：{activity or '未知'}；"
                    f"可见文案：{visible_labels}。"
                )

            tap_result = await self._action_tool.execute(
                action="tap",
                x=close_element.center[0],
                y=close_element.center[1],
            )
            if not _is_success("action_tool", "tap", tap_result):
                return f"启动页收敛失败：{tap_result}"
            await asyncio.sleep(1.0)

        try:
            parsed = await self._driver_manager.get_parsed_screen()
            activity = await self._driver_manager.get_current_activity()
        except Exception as e:
            return _describe_driver_crash("device_tool", e)
        if _looks_like_splash_screen(parsed, activity):
            visible_labels = "、".join(parsed.visible_labels(limit=6)) or "无明显文案"
            return (
                "启动页收敛后仍停留在启动/广告页。"
                f"当前 Activity：{activity or '未知'}；"
                f"可见文案：{visible_labels}。"
            )
        return "已自动关闭启动页干扰元素并进入业务页面。"

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


def _select_cases(
    payload: dict[str, Any],
    case_ids: list[str],
    max_cases: int | None,
    exclude_types: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """按批量条件选取待执行的用例列表，返回 (选中用例, 跳过的 case_id)。

    - 显式给定 ``case_ids`` 时，按给定顺序精确匹配；未命中的 id 记入跳过列表，
      不再回退到数量/类型过滤（调用方已经明确指定了要跑哪些用例）。
    - 否则按 JSON 声明顺序取全部用例，先按 ``exclude_types`` 过滤掉指定类型
      （如"回归测试"），再按 ``max_cases`` 截取前 N 条——即"前 N 条非排除类型
      的用例"，而不是"JSON 前 N 个位置里再筛掉排除类型"。
    """
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError("自动化 JSON 中未找到 cases。")

    if case_ids:
        by_id = {str(c.get("id", "")).strip(): c for c in cases}
        selected: list[dict[str, Any]] = []
        skipped: list[str] = []
        for cid in case_ids:
            case = by_id.get(cid.strip())
            if case is not None:
                selected.append(case)
            else:
                skipped.append(cid)
        if not selected:
            raise ValueError(f"未找到 case_ids={case_ids!r} 对应的任何自动化用例。")
        return selected, skipped

    normalized_excludes = {t.strip() for t in exclude_types if t.strip()}
    filtered = [c for c in cases if str(c.get("type", "")).strip() not in normalized_excludes]
    skipped = [
        str(c.get("id", "")).strip()
        for c in cases
        if str(c.get("type", "")).strip() in normalized_excludes
    ]
    if max_cases is not None and max_cases > 0:
        filtered = filtered[:max_cases]
    if not filtered:
        raise ValueError("按给定条件筛选后没有可执行的用例。")
    return filtered, skipped


def _is_success(tool_name: str, action_name: str, result: str) -> bool:
    if tool_name == "assertion_tool":
        return result.startswith("✅ PASS")
    failure_markers = ("错误", "失败", "未找到", "失效", "不可用")
    return not any(marker in result for marker in failure_markers)


# UiAutomator2 instrumentation 进程在设备侧崩溃时的特征字符串（Appium 代理报错
# 原文包含这段英文，不同 Appium/驱动版本措辞略有差异，做子串匹配即可）。
_INSTRUMENTATION_CRASH_MARKERS = (
    "instrumentation process is not running",
    "instrumentation process is not running (probably crashed)",
)


def _describe_driver_crash(tool_name: str, error: Exception) -> str:
    """把执行步骤时抛出的原始驱动异常转换成可读的步骤失败文案。

    踩过的坑：UiAutomator2 服务端在设备侧偶发崩溃（长时间连续操作后的环境
    抖动，与代码逻辑无关）时，Appium 对任意指令都会抛出未捕获的
    ``WebDriverException``；如果不在这里兜底捕获，异常会一路冒泡穿透
    `_run_case`/`execute_batch`，把整个批量执行（包括前面已经跑完、本该
    保留的 case 结果）一起打断。这里统一转换成普通的"步骤失败"字符串，让
    调用方按"这一条 case 失败"正常收尾、批量执行继续跑下一条——下一条
    case 开始前的 ``probe_session_alive()`` 探活会自动发现会话已死并重新
    连接，不需要在这里做重连尝试。
    """
    message = str(error)
    if any(marker in message for marker in _INSTRUMENTATION_CRASH_MARKERS):
        return (
            f"{tool_name} 执行失败：设备端 UiAutomator2 服务崩溃"
            "（instrumentation process crashed，环境层面的偶发抖动，非用例本身问题），"
            "本条用例判定失败；后续用例会在开始前自动探测并重新建立连接。"
        )
    return f"{tool_name} 执行失败：底层驱动异常：{message}"


_SPLASH_ACTIVITY_KEYWORDS = ("splash", "launcher")
_SPLASH_CLOSE_TEXTS = ("关闭", "跳过")
_SPLASH_CTA_TEXTS = ("点击前往详情页",)


def _looks_like_splash_screen(parsed, activity: str) -> bool:
    """根据 activity 与页面文案判断当前是否仍停留在启动页。"""
    normalized_activity = (activity or "").lower()
    if any(keyword in normalized_activity for keyword in _SPLASH_ACTIVITY_KEYWORDS):
        return True
    if _find_first_visible_text(parsed, _SPLASH_CTA_TEXTS) is not None:
        return True
    if _find_first_visible_text(parsed, _SPLASH_CLOSE_TEXTS) is not None:
        return True
    return False


def _find_first_visible_text(parsed, candidates: tuple[str, ...]):
    """在当前页面上查找第一个匹配候选文案的可见元素。"""
    for candidate in candidates:
        element = parsed.find_by_text(candidate)
        if element is not None and element.is_visible:
            return element
    return None


def _build_metadata(request_id: str) -> dict:
    metadata: dict[str, str] = {}
    if request_id:
        metadata["request_id"] = request_id
    return metadata
