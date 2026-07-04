"""ActionTool 滑动距离与底部悬浮层自动微调（nudge）单元测试。

回归背景：
1. ``_perform_single_swipe`` 之前接收 ``distance`` 参数但完全忽略它，
   只能做固定比例的整屏滑动，无法支撑"小幅微调"这类场景。
2. 自动化用例常见写法是"滚动到模块标题可见即视为找到模块"，但模块内部
   贴近屏幕底部的卡片/按钮可能仍被底部导航栏悬浮层盖住一部分；直接按
   解析出的坐标点击容易点空或误触导航栏。``_resolve_coords`` 现在会在
   探测到目标元素被遮挡时，自动小幅继续滚动直到目标清出遮挡区域。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.agent.tools.mobile.action_tool import ActionTool
from src.mobile.screen_parser import ParsedScreen, UIElement


def _el(text: str, bounds: list[int], clickable: bool = False) -> UIElement:
    return UIElement(
        text=text,
        resource_id="",
        class_name="android.widget.TextView",
        content_desc="",
        bounds=bounds,
        clickable=clickable,
        enabled=True,
        checkable=False,
        checked=False,
        focusable=False,
    )


def _make_tool(driver_manager) -> ActionTool:
    return ActionTool(driver_manager=driver_manager, page_cache=MagicMock())


class TestPerformSingleSwipeDistance(unittest.IsolatedAsyncioTestCase):
    async def test_default_distance_matches_original_half_screen_behavior(self):
        # distance 为 None 时必须保持原有的 3/4 -> 1/4 整屏滑动行为不变。
        mgr = MagicMock()
        mgr._driver = None  # 触发默认分辨率 1080x2340
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        await tool._perform_single_swipe(direction="down", distance=None, duration_ms=800)

        mgr.swipe.assert_awaited_once()
        sx, sy, ex, ey, duration = mgr.swipe.await_args.args
        self.assertEqual((sx, sy), (540, int(2340 * 0.75)))
        self.assertEqual((ex, ey), (540, int(2340 * 0.25)))

    async def test_custom_distance_produces_smaller_swipe_from_same_anchor(self):
        # 自定义 distance 应该从同一起点出发、只收缩终点，方便做小幅微调。
        mgr = MagicMock()
        mgr._driver = None
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        await tool._perform_single_swipe(direction="down", distance=200, duration_ms=400)

        sx, sy, ex, ey, duration = mgr.swipe.await_args.args
        self.assertEqual((sx, sy), (540, int(2340 * 0.75)))
        self.assertEqual(ey, sy - 200)


class TestNudgeIntoView(unittest.IsolatedAsyncioTestCase):
    async def test_tap_auto_scrolls_when_target_occluded_by_bottom_overlay(self):
        # 底部导航栏悬浮层固定不动；"加追"按钮起初有一部分落在悬浮层区域，
        # 一次小幅下滑后随内容一起上移，完全清出遮挡区域。
        bottom_bar = _el("", [0, 1546, 1080, 1792])
        title_before = _el("每日更新", [36, 1278, 228, 1335])
        button_occluded = _el("加追", [400, 1600, 600, 1650], clickable=True)
        screen_occluded = ParsedScreen(
            elements=[bottom_bar, title_before, button_occluded]
        )

        title_after = _el("每日更新", [36, 1128, 228, 1185])
        button_clear = _el("加追", [400, 1450, 600, 1500], clickable=True)
        screen_clear = ParsedScreen(elements=[bottom_bar, title_after, button_clear])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(side_effect=[screen_occluded, screen_clear])
        mgr.swipe = AsyncMock()
        mgr.tap = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool.execute(action="tap", target="加追", target_type="text")

        # 应先触发一次小幅下滑（nudge），再点击清出遮挡区域后的坐标。
        mgr.swipe.assert_awaited_once()
        mgr.tap.assert_awaited_once_with(*button_clear.center)
        self.assertIn("已点击坐标", result)

    async def test_tap_does_not_nudge_when_target_already_visible(self):
        # 没有遮挡时不应触发任何额外滑动，保持零开销。
        title = _el("每日更新", [36, 1278, 228, 1335])
        button = _el("加追", [400, 1400, 600, 1450], clickable=True)
        screen = ParsedScreen(elements=[title, button])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(return_value=screen)
        mgr.swipe = AsyncMock()
        mgr.tap = AsyncMock()
        tool = _make_tool(mgr)

        await tool.execute(action="tap", target="加追", target_type="text")

        mgr.swipe.assert_not_awaited()
        mgr.tap.assert_awaited_once_with(*button.center)


class TestScrollUntilConditionSettles(unittest.IsolatedAsyncioTestCase):
    async def test_scroll_continues_nudging_until_module_fully_clear(self):
        # 复现真实报错场景：滚动一次后"每日更新"标题已可见（满足 stop_condition），
        # 但模块内的作品卡片行仍有一部分压在底部导航栏下面。scroll 不应该在
        # 标题一出现就立刻返回，而要继续小幅滚动直到卡片本体也清出遮挡区域。
        bottom_bar = _el("", [0, 1546, 1080, 1792])
        title_stage1 = _el("每日更新", [36, 1278, 228, 1335])
        item_stage1 = _el("09:00\n更新", [48, 1479, 168, 1599])
        screen_stage1 = ParsedScreen(elements=[bottom_bar, title_stage1, item_stage1])

        title_stage2 = _el("每日更新", [36, 1009, 228, 1066])
        item_stage2 = _el("09:00\n更新", [48, 1210, 168, 1330])
        screen_stage2 = ParsedScreen(elements=[bottom_bar, title_stage2, item_stage2])

        mgr = MagicMock()
        mgr._driver = None
        # 第一次 get_parsed_screen 用于判断 stop_condition 是否命中（命中）；
        # 第二次用于 _settle_module 检测锚点下方是否仍被遮挡（仍遮挡，触发一次
        # nudge）；第三次是 nudge 之后重新检查（已清除遮挡，结束微调）。
        mgr.get_parsed_screen = AsyncMock(
            side_effect=[screen_stage1, screen_stage1, screen_stage2]
        )
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool._scroll_until_condition(
            direction="down",
            distance=None,
            duration_ms=400,
            max_swipes=5,
            stop_condition="text=每日更新",
        )

        self.assertIn("已满足停止条件", result)
        # 一次主滚动都没做（第一次就命中标题），但应该额外做了一次微调 nudge。
        mgr.swipe.assert_awaited_once()
        # 返回文案要如实体现微调次数，避免像批量执行那次一样"共滑动 0 次"
        # 却在背后偷偷多滑了几屏，导致后续 case 定位不到目标模块。
        self.assertIn("含微调 1 次", result)
        # 滑动距离应按骑线元素的实际重叠量精确计算（重叠 53px + 安全余量），
        # 而不是固定滑动屏幕高度 15% 的经验值——固定比例在长列表页面里
        # 会导致"清干净一个模块，又把下一个模块的内容带出来贴到边界上"，
        # 反复触发、越滑越远，最终把目标模块本身滑出屏幕。
        sx, sy, ex, ey, _duration = mgr.swipe.await_args.args
        overhang = 1599 - 1546  # item_stage1.y2 - boundary
        expected_distance = overhang + max(60, int(2340 * 0.03))
        self.assertEqual(sy - ey, expected_distance)

    async def test_scroll_does_not_nudge_for_distant_next_module_straddling_boundary(self):
        # 复现真实的"永不收敛"报错场景：即使排除了完全位于边界之下的内容，
        # 只要页面还能继续下滚，就总会有下一张完全不相关的卡片持续滑入视野、
        # 贴着遮挡边界线（骑线），而不是锚点自身模块被切了一半。这次的
        # next_card 满足"骑线"条件（y1 < boundary <= y2），但距离锚点已经
        # 超出一个模块合理的高度预算，应被视为下一模块的内容而非本模块被遮挡，
        # 否则会导致 _settle_module 永远判定为"仍被遮挡"，不断继续下滑，
        # 直至把锚点标题本身也滑出屏幕（批量执行 003~006 全部失败的根因）。
        bottom_bar = _el("", [0, 1546, 1080, 1792])
        title = _el("每日更新", [36, 200, 228, 257])
        item = _el("09:00\n更新", [48, 400, 168, 520])
        next_card = _el("日漫新作卡片", [36, 1400, 228, 1600])
        screen = ParsedScreen(elements=[bottom_bar, title, item, next_card])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(return_value=screen)
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool._scroll_until_condition(
            direction="down",
            distance=None,
            duration_ms=400,
            max_swipes=5,
            stop_condition="text=每日更新",
        )

        self.assertIn("已满足停止条件", result)
        mgr.swipe.assert_not_awaited()

    async def test_scroll_does_not_nudge_for_next_floor_content_below_boundary(self):
        # 回归场景：批量执行连续跑多条 case 时，"每日更新"命中后，屏幕下方
        # 还残留着下一个楼层（另一个完全独立的模块）已经紧贴在悬浮层上方
        # 的内容——这部分内容本来就该在那里、还未轮到，不代表当前模块被
        # 遮挡切了一半。旧版判定只看"y1 >= 锚点.y1"，会把它也算作遮挡，
        # 导致每条 case 都被误触发额外下滑，多条 case 累积后把目标模块彻底
        # 滑出屏幕（用户反馈"003 case 找不到元素"的根因）。修复后只应关注
        # "骑跨在遮挡边界上"的元素，下一楼层完整位于边界之下则不应触发。
        bottom_bar = _el("", [0, 1546, 1080, 1792])
        title = _el("每日更新", [36, 1278, 228, 1335])
        item = _el("09:00\n更新", [48, 1340, 168, 1460])
        next_floor = _el("日漫新作", [36, 1560, 228, 1650])
        screen = ParsedScreen(elements=[bottom_bar, title, item, next_floor])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(return_value=screen)
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool._scroll_until_condition(
            direction="down",
            distance=None,
            duration_ms=400,
            max_swipes=5,
            stop_condition="text=每日更新",
        )

        self.assertIn("已满足停止条件", result)
        mgr.swipe.assert_not_awaited()

    async def test_scroll_stops_immediately_when_module_already_fully_visible(self):
        # 没有遮挡时，命中停止条件应立即返回，不做任何额外滑动。
        title = _el("每日更新", [36, 1278, 228, 1335])
        item = _el("09:00\n更新", [48, 1340, 168, 1400])
        screen = ParsedScreen(elements=[title, item])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(return_value=screen)
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool._scroll_until_condition(
            direction="down",
            distance=None,
            duration_ms=400,
            max_swipes=5,
            stop_condition="text=每日更新",
        )

        self.assertIn("已满足停止条件", result)
        mgr.swipe.assert_not_awaited()

    async def test_scroll_retries_first_check_before_swiping_when_transition_not_settled(self):
        # 复现真实报错场景：back() 返回后立即调用 scroll，Activity 转场动画/
        # 列表布局还没完全稳定，第一次检查（滑动前）误判"目标不可见"。此时
        # 目标其实已经正常展示在屏幕上，不应该立即判定为"需要滑动"并执行
        # 若干次滑动——那会把已经就位的模块滑到很远的地方，且方向不可逆。
        # 应该先在原地短暂重试几次，确认转场/渲染稳定后再决定要不要滑动。
        empty_screen = ParsedScreen(elements=[])
        title = _el("每日更新", [36, 1278, 228, 1335])
        settled_screen = ParsedScreen(elements=[title])

        mgr = MagicMock()
        mgr._driver = None
        # 后续调用（如 _settle_module 内部再次读取屏幕）沿用同一个已稳定的画面。
        mgr.get_parsed_screen = AsyncMock(
            side_effect=[empty_screen, settled_screen, settled_screen]
        )
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool._scroll_until_condition(
            direction="down",
            distance=None,
            duration_ms=400,
            max_swipes=8,
            stop_condition="text=每日更新",
        )

        self.assertIn("已满足停止条件", result)
        self.assertIn("共滑动 0 次", result)
        mgr.swipe.assert_not_awaited()

    async def test_scroll_still_swipes_when_target_genuinely_not_on_screen(self):
        # 确认重试只是"给转场动画一点时间"，如果重试用完仍然找不到，
        # 该滑动还是要正常滑动，不能因为加了重试就完全不滑了。
        empty_screen = ParsedScreen(elements=[])
        title = _el("每日更新", [36, 1278, 228, 1335])

        settled_screen = ParsedScreen(elements=[title])
        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(
            side_effect=[empty_screen, empty_screen, empty_screen, settled_screen, settled_screen]
        )
        mgr.swipe = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool._scroll_until_condition(
            direction="down",
            distance=None,
            duration_ms=400,
            max_swipes=8,
            stop_condition="text=每日更新",
        )

        self.assertIn("已满足停止条件", result)
        mgr.swipe.assert_awaited_once()


class TestTapRetryAndIdIndex(unittest.IsolatedAsyncioTestCase):
    async def test_tap_retries_when_target_not_yet_rendered(self):
        # 复现真实报错场景："加追"按钮的选中状态需要额外接口返回后才完成
        # 渲染，页面标题已经出现在无障碍树里，但按钮文字还没写入。第一次
        # 查找落空不应该立即判定失败，应该短暂重试。
        empty_screen = ParsedScreen(elements=[_el("每日更新", [36, 1278, 228, 1335])])
        button = _el("加追", [400, 1400, 600, 1450], clickable=True)
        loaded_screen = ParsedScreen(elements=[_el("每日更新", [36, 1278, 228, 1335]), button])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(side_effect=[empty_screen, loaded_screen])
        mgr.tap = AsyncMock()
        tool = _make_tool(mgr)

        result = await tool.execute(action="tap", target="加追", target_type="text")

        self.assertIn("已点击坐标", result)
        mgr.tap.assert_awaited_once_with(*button.center)

    async def test_tap_by_id_supports_index_for_repeated_resource_id(self):
        # RecyclerView 里同一 resource-id 会在每张卡片上重复出现（如封面控件），
        # target_type=id 需要配合 index 才能精确定位"第 N 张卡片的封面"，
        # 而不是被迫用泛化的 class 通配去猜、点到无关元素。
        cover1 = UIElement(
            text="", resource_id="com.iqiyi.acg:id/iv_card_410_cover",
            class_name="android.view.View", content_desc="",
            bounds=[36, 1090, 356, 1516], clickable=False, enabled=True,
            checkable=False, checked=False, focusable=False,
        )
        cover2 = UIElement(
            text="", resource_id="com.iqiyi.acg:id/iv_card_410_cover",
            class_name="android.view.View", content_desc="",
            bounds=[380, 1090, 700, 1516], clickable=False, enabled=True,
            checkable=False, checked=False, focusable=False,
        )
        screen = ParsedScreen(elements=[cover1, cover2])

        mgr = MagicMock()
        mgr._driver = None
        mgr.get_parsed_screen = AsyncMock(return_value=screen)
        mgr.tap = AsyncMock()
        tool = _make_tool(mgr)

        await tool.execute(
            action="tap", target="iv_card_410_cover", target_type="id", index=1
        )

        mgr.tap.assert_awaited_once_with(*cover2.center)


class TestSwipeRetryOnTransientFailure(unittest.IsolatedAsyncioTestCase):
    async def test_swipe_retries_once_after_transient_error_and_succeeds(self):
        # 复现真实报错场景：UiAutomator2 偶发抛出 INJECT_EVENTS 权限异常，
        # 属于设备/环境层面的瞬时抖动，不应该让整条 case 直接判失败——
        # 短暂重试一次即可恢复。
        mgr = MagicMock()
        mgr._driver = None
        mgr.swipe = AsyncMock(
            side_effect=[RuntimeError("INJECT_EVENTS permission"), None]
        )
        tool = _make_tool(mgr)

        result = await tool._perform_single_swipe(
            direction="up", distance=None, duration_ms=400
        )

        self.assertIn("已向 up 滑动", result)
        self.assertEqual(mgr.swipe.await_count, 2)

    async def test_swipe_reports_failure_after_retry_exhausted(self):
        # 重试次数用完仍然失败时，要如实透出原始异常信息，而不是静默吞掉。
        mgr = MagicMock()
        mgr._driver = None
        mgr.swipe = AsyncMock(side_effect=RuntimeError("INJECT_EVENTS permission"))
        tool = _make_tool(mgr)

        result = await tool._perform_single_swipe(
            direction="up", distance=None, duration_ms=400
        )

        self.assertIn("swipe 失败", result)
        self.assertEqual(mgr.swipe.await_count, ActionTool._SWIPE_RETRY_ATTEMPTS + 1)


if __name__ == "__main__":
    unittest.main()
