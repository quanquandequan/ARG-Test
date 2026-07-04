"""AssertionTool：在测试步骤后验证 UI 状态。

断言不会修改设备状态，也不会使缓存失效。
它们会返回 pass/fail 结果以及描述性消息，供 Agent 纳入测试执行报告。

支持的操作：
  assert_text           当前屏幕中可见指定文字
  assert_not_text       当前屏幕中不可见指定文字（反向断言）
  assert_element        按 text/id/desc 识别的元素存在且可见
  assert_clickable      元素可见且可点击
  assert_page           当前 activity 匹配预期页面 / activity 名称
  assert_not_page       当前 activity 不再是指定的页面（反向断言，用于验证
                        "已经离开原页面"，适合跳转目标是同 Activity 内覆盖层/
                        半屏浮层等无法提前得知具体 Activity 名的场景）
  assert_checked        复选框 / 单选按钮处于选中状态
"""

from __future__ import annotations

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.mobile.driver import AppiumDriverManager

logger = get_logger(__name__)

_PASS = "✅ PASS"
_FAIL = "❌ FAIL"


class AssertionTool(BaseTool):
    name = "assertion_tool"
    description = (
        "验证 Android 页面的 UI 状态。"
        "支持断言：assert_text（文字可见）、assert_not_text（文字不可见）、"
        "assert_element（元素存在）、assert_clickable（元素可点击）、"
        "assert_page（当前页面匹配）、assert_not_page（当前页面已不是指定页面，"
        "用于验证跳转到了未知/易变的详情页）、assert_checked（复选框已勾选）。"
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
                    "assert_not_page",
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
            "element_class": {
                "type": "string",
                "description": "元素 class name（assert_element / assert_clickable 备选匹配）",
            },
            "page": {
                "type": "string",
                "description": (
                    "期望的 Activity 名称或包含关键字"
                    "（assert_page 使用；assert_not_page 时表示要离开的原页面）"
                ),
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

    # ── 入口 ─────────────────────────────────────────────────────────────────

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
            return await self._assert_page(should_match=True, **kwargs)
        if action == "assert_not_page":
            return await self._assert_page(should_match=False, **kwargs)
        if action == "assert_checked":
            return await self._assert_checked(**kwargs)

        return (
            f"未知断言：{action}。"
            "支持：assert_text / assert_not_text / assert_element / "
            "assert_clickable / assert_page / assert_not_page / assert_checked"
        )

    # ── 断言 ─────────────────────────────────────────────────────────────────

    async def _get_parsed(self):
        return await self._mgr.get_parsed_screen()

    async def _resolve_target(
        self,
        parsed,
        element_id: str = "",
        element_text: str = "",
        element_class: str = "",
    ):
        """共享元素查找：先尝试 resource-id，再尝试文本和 class name。"""
        el = None
        if element_id:
            el = parsed.find_by_resource_id(element_id)
        if el is None and element_text:
            el = parsed.find_by_text(element_text)
        if el is None and element_class:
            el = parsed.find_by_class_name(element_class)
        return el

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
        element_class: str = "",
        require_clickable: bool = False,
        **_,
    ) -> str:
        if not element_id and not element_text and not element_class:
            return f"{_FAIL} 必须提供 element_id、element_text 或 element_class 参数。"

        parsed = await self._get_parsed()
        el = await self._resolve_target(parsed, element_id, element_text, element_class)
        identifier = element_id or element_text or element_class

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
        should_match: bool = True,
        **_,
    ) -> str:
        """校验当前 Activity 是否（不）匹配指定页面。

        ``should_match=False`` 对应 ``assert_not_page``：只验证"已经离开了
        指定页面"，不要求知道跳转目标的确切 Activity 名——很多点击封面/
        进入详情这类场景，落地页可能是覆盖在同一 Activity 上的半屏浮层，
        或是一个事先不知道类名的新 Activity，用中文业务名称去匹配英文
        Activity 类名本来就不可能命中，这种场景改用 assert_not_page 更可靠。
        """
        if not page:
            return f"{_FAIL} 必须提供 page 参数（Activity 名或关键字）。"

        activity = await self._mgr.get_current_activity()
        package = await self._mgr.get_current_package()
        full = f"{package}/{activity}"

        if exact_match:
            matched = page == activity or page == full
        else:
            matched = page in activity or page in full

        if should_match:
            if matched:
                return f'{_PASS} 当前页面匹配："{page}"（Activity：{activity}）'
            return f'{_FAIL} 当前页面不匹配：期望 "{page}"，实际 Activity：{activity}'

        if not matched:
            return f'{_PASS} 已离开页面："{page}"（当前 Activity：{activity}）'
        return f'{_FAIL} 仍停留在页面："{page}"（当前 Activity：{activity}）'

    async def _assert_checked(
        self,
        element_id: str = "",
        element_text: str = "",
        **_,
    ) -> str:
        if not element_id and not element_text:
            return f"{_FAIL} 必须提供 element_id 或 element_text 参数。"

        parsed = await self._get_parsed()
        el = await self._resolve_target(parsed, element_id, element_text)
        identifier = element_id or element_text
        if el is None:
            return f"{_FAIL} 未找到元素：{identifier}"
        if not el.checkable:
            return f"{_FAIL} 元素不支持选中状态：{identifier}（class={el.class_name}）"
        if el.checked:
            return f"{_PASS} 元素已勾选：{identifier}"
        return f"{_FAIL} 元素未勾选：{identifier}"
