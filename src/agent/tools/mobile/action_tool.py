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

    # `_settle_module` 判断"骑线元素是否属于锚点自身模块"时的高度预算，
    # 以屏幕高度的比例表示；超出该范围的骑线内容视为下一个模块，不纳入判断。
    _MODULE_HEIGHT_RATIO = 0.35

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

    # 目标元素首次未命中时的重试次数与间隔：应对"关注/加追"这类按钮状态
    # 需要额外接口返回后才完成渲染的场景——页面标题/卡片本身已经出现在
    # 无障碍树里，但按钮文字可能还要再等一两帧才会被写入，直接判定"未找到"
    # 会造成偶发性的误报失败。
    _LOOKUP_RETRY_ATTEMPTS = 2
    _LOOKUP_RETRY_INTERVAL_S = 0.5

    # swipe 底层调用失败（如 UiAutomator2 偶发 INJECT_EVENTS 权限异常）时的重试次数与间隔
    _SWIPE_RETRY_ATTEMPTS = 1
    _SWIPE_RETRY_INTERVAL_S = 0.5

    async def _resolve_coords(
        self,
        target: str = "",
        target_type: str = "text",
        x: float | None = None,
        y: float | None = None,
        index: int = 0,
    ) -> tuple[int, int] | None:
        """解析元素坐标；未找到时返回 None。

        找到目标元素后，如果它被检测到的底部悬浮层（如常驻 Tab 导航栏）
        部分或整体遮挡，会先小幅继续向下滚动把目标"顶"出遮挡区域再返回坐标。
        这类场景常见于：滚动的停止条件只判断了模块标题可见，但模块内部
        贴近屏幕底部的卡片/按钮仍被底部导航栏盖住一部分，直接按解析出的
        坐标点击容易点空或误触导航栏。
        """
        if target:
            element = None
            for attempt in range(self._LOOKUP_RETRY_ATTEMPTS + 1):
                parsed = await self._mgr.get_parsed_screen()
                element = self._find_target_element(
                    parsed,
                    target=target,
                    target_type=target_type,
                    index=index,
                )
                if element is not None or attempt == self._LOOKUP_RETRY_ATTEMPTS:
                    break
                await asyncio.sleep(self._LOOKUP_RETRY_INTERVAL_S)
            if element is not None and parsed.is_occluded(element):
                nudged = await self._nudge_into_view(
                    target=target, target_type=target_type, index=index,
                )
                if nudged is not None:
                    element = nudged
            if element is not None:
                return element.center

        if x is not None and y is not None:
            return int(x), int(y)

        return None

    async def _nudge_into_view(
        self,
        target: str,
        target_type: str,
        index: int,
        max_nudges: int = 4,
    ) -> UIElement | None:
        """当目标元素被底部悬浮层部分遮挡时，小幅继续滚动直到清出遮挡区域。

        只在探测到遮挡时才触发，正常场景零开销；每次只滑动一小段距离
        （屏幕高度的 15%），避免因为单次滑动过量把目标重新滑出屏幕上方，
        或滑过头触发下一屏内容加载。

        ``max_nudges`` 与 ``_settle_module`` 保持一致取 4（而非曾经的 3）：
        两者承担的是同一类"清除底部悬浮层遮挡"职责，卡片内部常是
        "封面图 + 标题/副标题 + 按钮"纵向堆叠的多层结构，每次微调可能只把
        当前骑线的那一层滑出遮挡区，随即下一层又顶上边界线，需要多来几次
        才能把最底下的按钮（如"加追"）完全滑出遮挡区；每次检测到已清出
        遮挡区就会立即提前返回，调高上限不会拖慢正常场景。
        """
        w, h = await self._get_screen_size()
        nudge_distance = max(1, int(h * 0.15))
        element = None
        for _ in range(max_nudges):
            swipe_result = await self._perform_single_swipe(
                direction="down", distance=nudge_distance, duration_ms=400
            )
            if swipe_result.startswith("swipe 失败") or swipe_result.startswith("未知滑动方向"):
                return element
            self._cache.invalidate()
            parsed = await self._mgr.get_parsed_screen()
            element = self._find_target_element(
                parsed, target=target, target_type=target_type, index=index
            )
            if element is not None and not parsed.is_occluded(element):
                return element
        return element

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
            # 支持 index：同一 resource-id 常见于 RecyclerView 里重复出现的卡片
            # 组件（如每张卡片自己的封面控件），不能永远只取第一个匹配。
            element = parsed.find_by_resource_id(normalized_target, index=index)
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

        ``distance`` 为 None 时使用默认比例（纵向 1/2 屏高、横向屏宽减安全
        边距），起点固定不变，只收缩终点——这样传入更小的 distance 可以做
        "微调"式的小幅滑动（例如 ``_nudge_into_view`` 用它清除底部悬浮层遮挡），
        而不会改变默认滑动的既有行为。
        """
        w, h = await self._get_screen_size()
        margin_x = int(w * _SWIPE_MARGIN)
        mid_x = w // 2

        direction = direction.lower()

        # direction 表示页面浏览方向，不是手指移动方向。
        # down = 向下浏览（手指上滑）：从 3/4 处滑到 1/4 处
        # up   = 向上浏览（手指下滑）：从 1/4 处滑到 3/4 处
        if direction == "down":
            span = distance if distance is not None else int(h * 0.5)
            span = max(1, min(span, int(h * 0.5)))
            sx, sy = mid_x, int(h * 0.75)
            ex, ey = mid_x, sy - span
        elif direction == "up":
            span = distance if distance is not None else int(h * 0.5)
            span = max(1, min(span, int(h * 0.5)))
            sx, sy = mid_x, int(h * 0.25)
            ex, ey = mid_x, sy + span
        elif direction == "left":
            span = distance if distance is not None else (w - 2 * margin_x)
            span = max(1, min(span, w - 2 * margin_x))
            sx, sy = margin_x, h // 2
            ex, ey = margin_x + span, h // 2
        elif direction == "right":
            span = distance if distance is not None else (w - 2 * margin_x)
            span = max(1, min(span, w - 2 * margin_x))
            sx, sy = w - margin_x, h // 2
            ex, ey = w - margin_x - span, h // 2
        else:
            return f"未知滑动方向：{direction}。支持：up / down / left / right"

        dist = abs(ey - sy) if direction in ("down", "up") else abs(ex - sx)
        # UiAutomator2 底层偶发 INJECT_EVENTS 权限异常（设备/环境层面的瞬时抖动，
        # 并非手势参数错误），一次性失败没必要直接判定整条 case 失败，做一次
        # 短暂重试通常就能恢复；仍失败才把原始异常信息透出。
        last_error: Exception | None = None
        for attempt in range(self._SWIPE_RETRY_ATTEMPTS + 1):
            try:
                await self._mgr.swipe(sx, sy, ex, ey, duration_ms)
                return f"已向 {direction} 滑动 {dist}px。"
            except Exception as e:
                last_error = e
                if attempt < self._SWIPE_RETRY_ATTEMPTS:
                    await asyncio.sleep(self._SWIPE_RETRY_INTERVAL_S)
        return f"swipe 失败：{last_error}"

    async def _scroll_until_condition(
        self,
        direction: str,
        distance: int | None,
        duration_ms: int,
        max_swipes: int | None,
        stop_condition: str,
    ) -> str:
        """滚动直到满足停止条件；最多滑动 10 次，避免陷入循环。

        命中停止条件（通常是模块标题文字）后，不会立即返回：模块标题先滚入
        可视区、但模块本体（卡片列表、按钮等）仍有一部分压在底部悬浮层
        （如常驻 Tab 导航栏）下面是很常见的情况——继续调用 ``_settle_module``
        做几次小幅微调，直到锚点下方的内容清出遮挡区域，避免后续断言/截图/
        点击看到的是不完整的模块。

        踩过的坑：紧跟在 `back`/`tap` 跳转这类会触发 Activity 切换动画的操作
        之后立即调用 scroll 时，第一次检查（``attempt == 0``，滑动前）很容易
        因为转场动画/列表布局还没完全稳定而误判"目标不可见"，进而按剧本
        做若干次"继续往下浏览"的滑动——如果此时目标其实已经正常展示在屏幕
        上，这些滑动就是纯粹的误伤，会把原本已经就位的模块滑到很远的地方，
        且这个方向的滑动无法在同一次调用里自我纠正。所以只在第一次检查上
        做几次短暂重试（``_LOOKUP_RETRY_ATTEMPTS``/``_LOOKUP_RETRY_INTERVAL_S``，
        与 tap 目标查找共用同一套重试参数），确认转场动画/渲染已经稳定之后
        再决定是否真的需要滑动，避免"没滑之前先冤枉它一下"。
        """
        swipes = max(1, min(int(max_swipes or 1), 10))
        for attempt in range(swipes + 1):
            anchor = await self._find_stop_condition_element(stop_condition)
            if anchor is None and attempt == 0:
                for _ in range(self._LOOKUP_RETRY_ATTEMPTS):
                    await asyncio.sleep(self._LOOKUP_RETRY_INTERVAL_S)
                    anchor = await self._find_stop_condition_element(stop_condition)
                    if anchor is not None:
                        break
            if anchor is not None:
                settle_nudges = await self._settle_module(anchor, direction, duration_ms)
                total = attempt + settle_nudges
                detail = f"已满足停止条件：{stop_condition}。共滑动 {total} 次"
                detail += f"（含微调 {settle_nudges} 次）。" if settle_nudges else "。"
                return detail
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

    async def _find_stop_condition_element(self, stop_condition: str) -> UIElement | None:
        """解析 ``text=...`` 或纯文本形式的停止条件，返回命中的可见元素。"""
        condition = stop_condition.strip()
        if not condition:
            return None
        target = (
            condition.split("=", 1)[1].strip().strip("\"'")
            if condition.startswith("text=")
            else condition
        )
        if not target:
            return None
        parsed = await self._mgr.get_parsed_screen()
        element = parsed.find_by_text(target)
        return element if element is not None and element.is_visible else None

    async def _settle_module(
        self,
        anchor: UIElement,
        direction: str,
        duration_ms: int,
        max_nudges: int = 4,
    ) -> int:
        """锚点命中后，若锚点自身模块的内容仍被底部悬浮层遮挡，做精确的一次性补偿滚动。

        只在探测到遮挡时才触发，正常场景零开销。返回实际执行的微调滑动次数，
        供调用方如实计入返回文案（避免这部分滚动完全"隐身"，此前曾出现日志
        显示"共滑动 0 次"、背后却偷偷多滑了好几屏的情况，导致问题难以排查）。

        踩过的坑：在可以无限下拉的长列表页面（RecyclerView）里，"锚点下方是否
        还有内容骑跨在遮挡边界上"这个条件几乎永远为真——因为只要页面还能继续
        往下滚，就总会有下一张完全不相关的卡片正在滑入视野、贴着边界线，而不是
        锚点自己模块的内容被切了一半。如果不加区分地反复检测、反复微调，会导致
        "越滑越远、永不收敛"，多个 case 连续执行时把目标模块连同锚点标题本身
        一起滑出屏幕（曾经在批量执行中把 003~006 全部拖垮的根因）。

        修复方式：
          1. 只认定"属于锚点自身模块"的骑线元素——限定在锚点往下
             ``_MODULE_HEIGHT_RATIO``（屏幕高度的一个比例）范围内，超出此范围
             视为下一个模块，不计入判断。
          2. 按骑线元素实际重叠量精确计算滑动距离（+ 少量安全余量），
             而不是固定滑动屏幕高度的某个百分比再反复试探。
          3. ``max_nudges`` 只作为兜底安全阀（防止计算误差导致一次没清除干净），
             不再是常态会被跑满的主循环。

        踩过的坑（二）：卡片内部往往是"封面图 + 标题/副标题 + 按钮"纵向堆叠的
        多层结构。第一次检测到的骑线元素可能只是封面图，按封面图的重叠量算出
        的距离刚好能把封面图滑出遮挡区，但紧随其后的标题/按钮此时又刚好顶上
        边界线、变成新的骑线元素，需要再来一次才能把按钮也滑出来。所以
        ``max_nudges`` 不能压得太低（之前设过 2，实测某些三层卡片两次不够、
        "加追"按钮始终留在遮挡区里）；由于范围已经被 ``_MODULE_HEIGHT_RATIO``
        限定在锚点自身模块内、且一旦检测不到骑线元素就会立即提前返回，多给
        几次机会不会引入"越滑越远"的旧问题，只会让同一张卡片的多层结构被
        完整地滑出遮挡区。
        """
        if not anchor.text:
            return 0
        nudges_done = 0
        for _ in range(max_nudges):
            parsed = await self._mgr.get_parsed_screen()
            current = parsed.find_by_text(anchor.text)
            if current is None:
                return nudges_done
            boundary = parsed.bottom_overlay_top()
            if boundary is None:
                return nudges_done
            _, screen_height = await self._get_screen_size()
            module_budget = int(screen_height * self._MODULE_HEIGHT_RATIO)
            straddling = [
                el
                for el in parsed.visible_elements()
                if current.bounds[1] <= el.bounds[1] < boundary <= el.bounds[3]
                and el.bounds[1] - current.bounds[1] <= module_budget
            ]
            if not straddling:
                return nudges_done
            overhang = max(el.bounds[3] - boundary for el in straddling)
            nudge_distance = overhang + max(60, int(screen_height * 0.03))
            await self._perform_single_swipe(
                direction=direction,
                distance=nudge_distance,
                duration_ms=duration_ms,
            )
            nudges_done += 1
        return nudges_done

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
