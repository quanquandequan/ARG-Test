"""AssertionTool — verify UI state after test steps.

Assertions do NOT modify the device state and do NOT invalidate the cache.
They return a pass/fail result with a descriptive message the Agent can
include in the test execution report.

Supported actions:
  assert_text           Text is visible somewhere on the current screen
  assert_not_text       Text is NOT visible (negative assertion)
  assert_element        Element identified by text/id/desc exists and is visible
  assert_clickable      Element is visible AND clickable
  assert_page           Current activity matches the expected page/activity name
  assert_checked        Checkbox / radio button is in checked state
"""

from __future__ import annotations

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.mobile.driver import AppiumDriverManager
from src.mobile.screen_parser import parse_page_source

logger = get_logger(__name__)

_PASS = "✅ PASS"
_FAIL = "❌ FAIL"


class AssertionTool(BaseTool):
    name = "assertion_tool"
    description = (
        "验证 Android 页面的 UI 状态。"
        "支持断言：assert_text（文字可见）、assert_not_text（文字不可见）、"
        "assert_element（元素存在）、assert_clickable（元素可点击）、"
        "assert_page（当前页面匹配）、assert_checked（复选框已勾选）。"
        "返回 PASS/FAIL 结果和描述，不修改设备状态。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "assert_text",
                    "assert_not_text",
                    "assert_element",
                    "assert_clickable",
                    "assert_page",
                    "assert_checked",
                ],
                "description": "断言类型",
            },
            "text": {
                "type": "string",
                "description": "期望的文字内容（assert_text / assert_not_text 使用）",
            },
            "element_id": {
                "type": "string",
                "description": "元素 resource-id（assert_element / assert_clickable 使用）",
            },
            "element_text": {
                "type": "string",
                "description": "元素文字（assert_element / assert_clickable 的备选匹配方式）",
            },
            "page": {
                "type": "string",
                "description": "期望的 Activity 名称或包含关键字（assert_page 使用）",
            },
            "exact_match": {
                "type": "boolean",
                "description": "是否严格匹配（默认 false，使用模糊包含匹配）",
            },
        },
        "required": ["action"],
    }

    def __init__(self, driver_manager: AppiumDriverManager) -> None:
        self._mgr = driver_manager

    # ── Entry point ───────────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs) -> str:  # type: ignore[override]
        if not self._mgr.is_connected():
            return f"{_FAIL} 设备未连接，请先调用 device_tool action=connect。"

        action = action.strip().lower()

        if action == "assert_text":
            return await self._assert_text(should_exist=True, **kwargs)
        if action == "assert_not_text":
            return await self._assert_text(should_exist=False, **kwargs)
        if action == "assert_element":
            return await self._assert_element(require_clickable=False, **kwargs)
        if action == "assert_clickable":
            return await self._assert_element(require_clickable=True, **kwargs)
        if action == "assert_page":
            return await self._assert_page(**kwargs)
        if action == "assert_checked":
            return await self._assert_checked(**kwargs)

        return (
            f"未知断言：{action}。"
            "支持：assert_text / assert_not_text / assert_element / "
            "assert_clickable / assert_page / assert_checked"
        )

    # ── Assertions ────────────────────────────────────────────────────────────

    async def _get_parsed(self):
        xml = await self._mgr.get_page_source()
        return parse_page_source(xml)

    async def _assert_text(
        self,
        text: str = "",
        exact_match: bool = False,
        should_exist: bool = True,
        **_,
    ) -> str:
        if not text:
            return f"{_FAIL} 必须提供 text 参数。"

        parsed = await self._get_parsed()
        found = parsed.find_by_text(text, exact=exact_match)
        visible = found is not None and found.is_visible

        if should_exist:
            if visible:
                return f'{_PASS} 页面中找到文字："{text}"'
            return f'{_FAIL} 页面中未找到文字："{text}"'
        else:
            if not visible:
                return f'{_PASS} 页面中确认不存在文字："{text}"'
            return f'{_FAIL} 页面中意外出现文字："{text}"'

    async def _assert_element(
        self,
        element_id: str = "",
        element_text: str = "",
        require_clickable: bool = False,
        **_,
    ) -> str:
        if not element_id and not element_text:
            return f"{_FAIL} 必须提供 element_id 或 element_text 参数。"

        parsed = await self._get_parsed()
        el = None
        if element_id:
            el = parsed.find_by_resource_id(element_id)
        if el is None and element_text:
            el = parsed.find_by_text(element_text)

        identifier = element_id or element_text

        if el is None or not el.is_visible:
            return f"{_FAIL} 未找到元素：{identifier}"

        if require_clickable and not el.clickable:
            return (
                f"{_FAIL} 元素存在但不可点击：{identifier} "
                f"（class={el.class_name}）"
            )

        qualifier = "（可点击）" if require_clickable else ""
        return f"{_PASS} 元素可见{qualifier}：{identifier}"

    async def _assert_page(
        self,
        page: str = "",
        exact_match: bool = False,
        **_,
    ) -> str:
        if not page:
            return f"{_FAIL} 必须提供 page 参数（Activity 名或关键字）。"

        activity = await self._mgr.get_current_activity()
        package = await self._mgr.get_current_package()
        full = f"{package}/{activity}"

        if exact_match:
            matched = page == activity or page == full
        else:
            matched = page in activity or page in full

        if matched:
            return f'{_PASS} 当前页面匹配："{page}"（Activity：{activity}）'
        return (
            f'{_FAIL} 当前页面不匹配：期望 "{page}"，'
            f"实际 Activity：{activity}"
        )

    async def _assert_checked(
        self,
        element_id: str = "",
        element_text: str = "",
        **_,
    ) -> str:
        if not element_id and not element_text:
            return f"{_FAIL} 必须提供 element_id 或 element_text 参数。"

        parsed = await self._get_parsed()
        el = None
        if element_id:
            el = parsed.find_by_resource_id(element_id)
        if el is None and element_text:
            el = parsed.find_by_text(element_text)

        identifier = element_id or element_text
        if el is None:
            return f"{_FAIL} 未找到元素：{identifier}"
        if not el.checkable:
            return f"{_FAIL} 元素不支持选中状态：{identifier}（class={el.class_name}）"
        if el.checked:
            return f"{_PASS} 元素已勾选：{identifier}"
        return f"{_FAIL} 元素未勾选：{identifier}"
