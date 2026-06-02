"""ActionTool — execute UI gestures on the Android device.

After every successful action, the PageCache is invalidated so the next
screen read reflects the updated state.

Supported actions:
  tap           Tap an element by text, resource-id, or coordinates
  long_press    Long-press an element (by text / id / coords)
  input_text    Type text into the currently focused field
  clear_text    Clear the focused text field
  swipe         Directional swipe (up/down/left/right) or custom coords
  scroll        Scroll in a direction (convenience wrapper around swipe)
  back          Press the Android Back button
  home          Press the Android Home button
"""

from __future__ import annotations

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.mobile.driver import AppiumDriverManager
from src.mobile.screen_parser import parse_page_source
from src.services.page_cache import PageCache

logger = get_logger(__name__)

# Default screen resolution assumptions for relative swipes
_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 2340
_SWIPE_MARGIN = 0.15   # stay 15% from edges to avoid triggering gestures
_SCROLL_DISTANCE = 0.4  # scroll 40% of screen height


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
                "enum": ["text", "id", "desc"],
                "description": "target 的匹配方式（默认 text）",
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

    # ── Entry point ───────────────────────────────────────────────────────────

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

        # Invalidate cache after any successful action
        self._cache.invalidate()
        return result

    # ── Action implementations ────────────────────────────────────────────────

    async def _resolve_coords(
        self,
        target: str = "",
        target_type: str = "text",
        x: float | None = None,
        y: float | None = None,
    ) -> tuple[int, int] | None:
        """Resolve element coordinates.  Returns None if nothing found."""
        if target:
            xml = await self._mgr.get_page_source()
            parsed = parse_page_source(xml)

            if target_type == "id":
                el = parsed.find_by_resource_id(target)
            elif target_type == "desc":
                # Treat content_desc field
                el = next(
                    (e for e in parsed.elements if target in e.content_desc),
                    None,
                )
            else:
                el = parsed.find_by_text(target)

            if el and el.is_visible:
                return el.center

        if x is not None and y is not None:
            return int(x), int(y)

        return None

    async def _tap(
        self,
        target: str = "",
        target_type: str = "text",
        x: float | None = None,
        y: float | None = None,
        **_,
    ) -> str:
        coords = await self._resolve_coords(target, target_type, x, y)
        if coords is None:
            hint = f'文字"{target}"' if target else f"坐标 ({x}, {y})"
            return f"未找到目标 {hint}，tap 操作未执行。"
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
        **_,
    ) -> str:
        coords = await self._resolve_coords(target, target_type, x, y)
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
        **_,
    ) -> str:
        w, h = _DEFAULT_WIDTH, _DEFAULT_HEIGHT
        margin_x = int(w * _SWIPE_MARGIN)
        mid_x = w // 2
        mid_y = h // 2

        direction = direction.lower()
        default_dist = int(h * _SCROLL_DISTANCE)
        dist = distance if distance else default_dist

        if direction == "up":
            sx, sy = mid_x, mid_y + dist // 2
            ex, ey = mid_x, mid_y - dist // 2
        elif direction == "down":
            sx, sy = mid_x, mid_y - dist // 2
            ex, ey = mid_x, mid_y + dist // 2
        elif direction == "left":
            sx, sy = w - margin_x, mid_y
            ex, ey = margin_x, mid_y
        elif direction == "right":
            sx, sy = margin_x, mid_y
            ex, ey = w - margin_x, mid_y
        else:
            return f"未知滑动方向：{direction}。支持：up / down / left / right"

        try:
            await self._mgr.swipe(sx, sy, ex, ey, duration_ms)
            return f"已向 {direction} 滑动 {dist}px。"
        except Exception as e:
            return f"swipe 失败：{e}"

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
