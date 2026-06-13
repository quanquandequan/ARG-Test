"""ActionTool：在 Android 设备上执行 UI 手势。

每次操作成功后都会使 PageCache 失效，确保下一次屏幕读取反映最新状态。

支持的操作：
  tap           按文本、resource-id、class name 或坐标点击元素
  long_press    按文本、id、class name 或坐标长按元素
  input_text    向当前聚焦字段输入文本
  clear_text    清空当前聚焦文本框
  swipe         按方向滑动（上/下/左/右）或自定义坐标
  scroll        按方向滚动（swipe 的便捷封装）
  back          按 Android 返回键
  home          按 Android Home 键
"""

from __future__ import annotations

import asyncio
import re

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.mobile.driver import AppiumDriverManager
from src.mobile.screen_parser import ParsedScreen, UIElement
from src.services.page_cache import PageCache

logger = get_logger(__name__)

# 相对滑动使用的默认屏幕分辨率假设
_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 2340
_SWIPE_MARGIN = 0.15   # 距离边缘保留 15%，避免触发系统手势
_SCROLL_DISTANCE = 0.4  # 滚动屏幕高度的 40%


class ActionTool(BaseTool):
    name = "action_tool"
    description = (
        "在 Android 设备上执行 UI 操作。"
        "支持：tap（点击）、long_press（长按）、input_text（输入文字）、"
        "clear_text（清空输入框）、swipe（滑动）、scroll（滚动）、"
        "back（返回）、home（返回桌面）。"
        "操作后自动使屏幕缓存失效，下次 screen_tool 将重新分析页面。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tap", "long_press", "input_text", "clear_text",
                         "swipe", "scroll", "back", "home"],
                "description": "要执行的操作",
            },
            "target": {
                "type": "string",
                "description": "点击/长按目标的文字或内容描述（tap/long_press 用）",
            },
            "target_type": {
                "type": "string",
                "enum": ["text", "id", "desc", "class"],
                "description": "target 的匹配方式（默认 text）",
            },
            "index": {
                "type": "integer",
                "description": "按 class 匹配时要选择的第几个可见元素（默认 0）",
            },
            "x": {"type": "number", "description": "横坐标（像素，优先级低于 target）"},
            "y": {"type": "number", "description": "纵坐标（像素，优先级低于 target）"},
            "text": {"type": "string", "description": "要输入的文字（input_text 用）"},
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "滑动/滚动方向（swipe/scroll 用）",
            },
            "distance": {
                "type": "integer",
                "description": "自定义滑动距离（像素），不填则使用默认值",
            },
            "max_swipes": {
                "type": "integer",
                "description": "滚动查找时的最大滑动次数，上限固定为 10",
            },
            "stop_condition": {
                "type": "string",
                "description": "滚动停止条件，例如 text=每日更新",
            },
            "duration_ms": {
                "type": "integer",
                "description": "手势持续时间（毫秒），默认 800",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        driver_manager: AppiumDriverManager,
        page_cache: PageCache,
    ) -> None:
        self._mgr = driver_manager
        self._cache = page_cache

    # ── 入口 ─────────────────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs) -> str:  # type: ignore[override]
        if not self._mgr.is_connected():
            return "错误：设备未连接，请先调用 device_tool action=connect。"

        action = action.strip().lower()

        if action == "tap":
            result = await self._tap(**kwargs)
        elif action == "long_press":
            result = await self._long_press(**kwargs)
        elif action == "input_text":
            result = await self._input_text(**kwargs)
        elif action == "clear_text":
            result = await self._clear_text()
        elif action in ("swipe", "scroll"):
            result = await self._swipe(**kwargs)
        elif action == "back":
            result = await self._back()
        elif action == "home":
            result = await self._home()
        else:
            return (
                f"未知操作：{action}。"
                "支持：tap / long_press / input_text / clear_text / swipe / scroll / back / home"
            )

        # 任意成功操作后使缓存失效
        self._cache.invalidate()
        return result

    # ── 操作实现 ─────────────────────────────────────────────────────────────

    async def _resolve_coords(
        self,
        target: str = "",
        target_type: str = "text",
        x: float | None = None,
        y: float | None = None,
        index: int = 0,
    ) -> tuple[int, int] | None:
        """解析元素坐标；未找到时返回 None。"""
        if target:
            parsed = await self._mgr.get_parsed_screen()
            element = self._find_target_element(
                parsed,
                target=target,
                target_type=target_type,
                index=index,
            )
            if element is not None:
                return element.center

        if x is not None and y is not None:
            return int(x), int(y)

        return None

    def _find_target_element(
        self,
        parsed: ParsedScreen,
        target: str,
        target_type: str,
        index: int = 0,
    ) -> UIElement | None:
        """按 target_type 在当前页面中解析目标元素。"""
        normalized_type = self._normalize_target_type(target_type)
        normalized_target = target.strip()

        if normalized_type == "id":
            element = parsed.find_by_resource_id(normalized_target)
            return element if element and element.is_visible else None

        if normalized_type == "desc":
            element = next(
                (
                    e
                    for e in parsed.visible_elements()
                    if normalized_target in e.content_desc
                ),
                None,
            )
            return element if element and element.is_visible else None

        if normalized_type == "class":
            class_name = self._extract_class_name(normalized_target)
            clickable_element = parsed.find_by_class_name(
                class_name,
                index=index,
                clickable_only=True,
            )
            if clickable_element is not None:
                return clickable_element
            return parsed.find_by_class_name(class_name, index=index, clickable_only=False)

        element = parsed.find_by_text(normalized_target)
        return element if element and element.is_visible else None

    def _normalize_target_type(self, target_type: str) -> str:
        """兼容生成器历史输出的 target_type 别名。"""
        normalized = (target_type or "text").strip().lower()
        alias_map = {
            "class_name": "class",
            "classname": "class",
            "locator": "class",
        }
        return alias_map.get(normalized, normalized)

    def _extract_class_name(self, target: str) -> str:
        """兼容 className=android.widget.ImageView 这类旧格式。"""
        match = re.match(r"^[A-Za-z_]+=(.+)$", target)
        if match:
            return match.group(1).strip()
        return target

    def _describe_lookup_failure(
        self,
        parsed: ParsedScreen,
        target: str,
        target_type: str,
    ) -> str:
        """生成失败时的补充诊断信息，便于定位页面状态。"""
        visible_labels = parsed.visible_labels(limit=6)
        labels_preview = "、".join(visible_labels) if visible_labels else "无明显文案"
        return (
            f'未找到 target="{target}"（target_type={target_type}）。'
            f"当前可见元素 {len(parsed.visible_elements())} 个，"
            f"可点击元素 {len(parsed.clickable_elements())} 个，"
            f"前几个可见文案：{labels_preview}。"
        )

    async def _tap(
        self,
        target: str = "",
        target_type: str = "text",
        x: float | None = None,
        y: float | None = None,
        index: int = 0,
        **_,
    ) -> str:
        coords = await self._resolve_coords(target, target_type, x, y, index=index)
        if coords is None:
            if target:
                parsed = await self._mgr.get_parsed_screen()
                failure_detail = self._describe_lookup_failure(parsed, target, target_type)
                return f"未找到目标 文字\"{target}\"，tap 操作未执行。{failure_detail}"
            return f"未找到目标 坐标 ({x}, {y})，tap 操作未执行。"
        cx, cy = coords
        try:
            await self._mgr.tap(cx, cy)
            return f"已点击坐标 ({cx}, {cy}){f'（目标：{target}）' if target else ''}。"
        except Exception as e:
            return f"tap 失败：{e}"

    async def _long_press(
        self,
        target: str = "",
        target_type: str = "text",
        x: float | None = None,
        y: float | None = None,
        duration_ms: int = 1000,
        index: int = 0,
        **_,
    ) -> str:
        coords = await self._resolve_coords(target, target_type, x, y, index=index)
        if coords is None:
            return f"未找到目标 {target or f'({x},{y})'}，long_press 未执行。"
        cx, cy = coords
        try:
            await self._mgr.long_press(cx, cy, duration_ms)
            return f"已长按坐标 ({cx}, {cy})，持续 {duration_ms} ms。"
        except Exception as e:
            return f"long_press 失败：{e}"

    async def _input_text(self, text: str = "", **_) -> str:
        if not text:
            return "错误：必须提供 text 参数。"
        try:
            await self._mgr.input_text(text)
            return f"已输入文字：{text!r}"
        except Exception as e:
            return f"input_text 失败：{e}"

    async def _clear_text(self) -> str:
        try:
            await self._mgr.clear_focused_field()
            return "已清空当前输入框内容。"
        except Exception as e:
            return f"clear_text 失败：{e}"

    async def _swipe(
        self,
        direction: str = "up",
        distance: int | None = None,
        x: float | None = None,
        y: float | None = None,
        duration_ms: int = 800,
        max_swipes: int | None = None,
        stop_condition: str = "",
        **_,
    ) -> str:
        if stop_condition.strip():
            return await self._scroll_until_condition(
                direction=direction,
                distance=distance,
                duration_ms=duration_ms,
                max_swipes=max_swipes,
                stop_condition=stop_condition,
            )

        return await self._perform_single_swipe(
            direction=direction,
            distance=distance,
            duration_ms=duration_ms,
        )

    async def _get_screen_size(self) -> tuple[int, int]:
        """从设备获取真实屏幕尺寸，回退到默认值。"""
        try:
            drv = self._mgr._driver
            if drv is not None:
                size = await asyncio.to_thread(lambda: drv.get_window_size())
                return int(size["width"]), int(size["height"])
        except Exception:
            pass
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT

    async def _perform_single_swipe(
        self,
        direction: str,
        distance: int | None,
        duration_ms: int,
    ) -> str:
        """执行一次滑动，direction 语义表示页面浏览方向。

        使用屏幕比例坐标（3/4 → 1/4）而非固定像素，
        兼容不同分辨率设备并避免被顶部 banner 拦截触摸事件。
        """
        w, h = await self._get_screen_size()
        margin_x = int(w * _SWIPE_MARGIN)
        mid_x = w // 2

        direction = direction.lower()

        # direction 表示页面浏览方向，不是手指移动方向。
        # down = 向下浏览（手指上滑）：从 3/4 处滑到 1/4 处
        # up   = 向上浏览（手指下滑）：从 1/4 处滑到 3/4 处
        if direction == "down":
            sx, sy = mid_x, int(h * 0.75)
            ex, ey = mid_x, int(h * 0.25)
        elif direction == "up":
            sx, sy = mid_x, int(h * 0.25)
            ex, ey = mid_x, int(h * 0.75)
        elif direction == "left":
            sx, sy = margin_x, h // 2
            ex, ey = w - margin_x, h // 2
        elif direction == "right":
            sx, sy = w - margin_x, h // 2
            ex, ey = margin_x, h // 2
        else:
            return f"未知滑动方向：{direction}。支持：up / down / left / right"

        dist = abs(ey - sy)
        try:
            await self._mgr.swipe(sx, sy, ex, ey, duration_ms)
            return f"已向 {direction} 滑动 {dist}px。"
        except Exception as e:
            return f"swipe 失败：{e}"

    async def _scroll_until_condition(
        self,
        direction: str,
        distance: int | None,
        duration_ms: int,
        max_swipes: int | None,
        stop_condition: str,
    ) -> str:
        """滚动直到满足停止条件；最多滑动 10 次，避免陷入循环。"""
        swipes = max(1, min(int(max_swipes or 1), 10))
        for attempt in range(swipes + 1):
            if await self._matches_stop_condition(stop_condition):
                return f"已满足停止条件：{stop_condition}。共滑动 {attempt} 次。"
            if attempt == swipes:
                break
            swipe_result = await self._perform_single_swipe(
                direction=direction,
                distance=distance,
                duration_ms=duration_ms,
            )
            if swipe_result.startswith("swipe 失败") or swipe_result.startswith("未知滑动方向"):
                return swipe_result

        return f"滑动 {swipes} 次后仍未满足停止条件：{stop_condition}。"

    async def _matches_stop_condition(self, stop_condition: str) -> bool:
        """支持 text=... 形式的可见文本停止条件。"""
        condition = stop_condition.strip()
        parsed = await self._mgr.get_parsed_screen()
        if condition.startswith("text="):
            target = condition.split("=", 1)[1].strip().strip("\"'")
            element = parsed.find_by_text(target)
            return element is not None and element.is_visible
        if condition:
            element = parsed.find_by_text(condition)
            return element is not None and element.is_visible
        return False

    async def _back(self) -> str:
        try:
            await self._mgr.press_back()
            return "已按下返回键。"
        except Exception as e:
            return f"back 失败：{e}"

    async def _home(self) -> str:
        try:
            await self._mgr.press_home()
            return "已按下 Home 键，返回桌面。"
        except Exception as e:
            return f"home 失败：{e}"
