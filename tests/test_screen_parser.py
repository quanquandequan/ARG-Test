"""screen_parser 底部悬浮层遮挡检测单元测试。

回归背景：很多 App 的底部导航栏是独立于内容滚动区域的固定悬浮层。
内容区域滚动到某个模块的标题可见后，模块内部贴近屏幕底部的卡片/按钮
可能仍有一部分被悬浮层盖住，但 UIAutomator2 汇报的 bounds 只是布局
坐标，``is_visible`` 判断不出这种视觉遮挡，导致按解析出的坐标点击时
点空或误触导航栏。``bottom_overlay_top`` / ``is_occluded`` 用启发式
识别这类悬浮层并判断目标元素是否被其遮挡。
"""

import unittest

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


class TestBottomOverlayDetection(unittest.TestCase):
    def test_detects_bottom_bar_and_flags_overlapping_item(self):
        # 模拟真实报错场景：屏幕高 1792，底部导航栏 [0,1546,1080,1792]，
        # 一个作品卡片项 [48,1479,168,1599] 有一部分落在导航栏区域内。
        bottom_bar = _el("", [0, 1546, 1080, 1792])
        module_title = _el("每日更新", [36, 1278, 228, 1335])
        overlapping_item = _el("09:00\n更新", [48, 1479, 168, 1599])
        fully_hidden_item = _el("加追", [48, 1600, 168, 1700])
        clear_item = _el("追番表", [885, 1283, 1044, 1329])

        screen = ParsedScreen(
            elements=[bottom_bar, module_title, overlapping_item, fully_hidden_item, clear_item]
        )

        boundary = screen.bottom_overlay_top()
        self.assertIsNotNone(boundary)
        self.assertEqual(boundary, 1546)

        self.assertTrue(screen.is_occluded(overlapping_item))
        self.assertTrue(screen.is_occluded(fully_hidden_item))
        self.assertFalse(screen.is_occluded(module_title))
        self.assertFalse(screen.is_occluded(clear_item))

    def test_no_bottom_overlay_detected_returns_none_and_never_occluded(self):
        # 没有任何贴底的宽幅容器时，不应误判存在遮挡层。
        title = _el("每日更新", [36, 1278, 228, 1335])
        screen = ParsedScreen(elements=[title])
        self.assertIsNone(screen.bottom_overlay_top())
        self.assertFalse(screen.is_occluded(title))

    def test_narrow_bottom_element_not_treated_as_overlay(self):
        # 贴底但横向跨度很窄的元素（例如一个小图标）不应被误判为悬浮导航栏。
        narrow = _el("", [500, 1700, 580, 1792])
        title = _el("每日更新", [36, 1278, 228, 1335])
        screen = ParsedScreen(elements=[narrow, title])
        self.assertIsNone(screen.bottom_overlay_top())


class TestFindByText(unittest.TestCase):
    def test_exact_match_preferred_over_unrelated_longer_text(self):
        # 复现真实报错场景：星期切换栏的日期缩写是单字"一"，页面上排在它
        # 前面的"换一批"刷新按钮文案里也包含"一"字。旧逻辑走子串包含匹配，
        # 会先命中"换一批"，导致点击了完全不相关的按钮却没有任何报错。
        refresh_button = _el("换一批", [729, 413, 837, 455])
        monday_tab = _el("一", [805, 648, 850, 732])
        screen = ParsedScreen(elements=[refresh_button, monday_tab])

        found = screen.find_by_text("一")

        self.assertIs(found, monday_tab)

    def test_fuzzy_fallback_still_works_when_no_exact_match(self):
        # 没有精确匹配时，仍要退化为子串包含匹配，保持原有的模糊查找能力。
        title = _el("追番表Card-每日更新", [36, 1278, 228, 1335])
        screen = ParsedScreen(elements=[title])

        found = screen.find_by_text("每日更新")

        self.assertIs(found, title)

    def test_exact_flag_disables_fuzzy_fallback(self):
        # exact=True 时不应该退化为子串匹配。
        title = _el("追番表Card-每日更新", [36, 1278, 228, 1335])
        screen = ParsedScreen(elements=[title])

        self.assertIsNone(screen.find_by_text("每日更新", exact=True))


class TestFindByResourceIdIndex(unittest.TestCase):
    def _el_with_id(self, resource_id: str, bounds: list[int]) -> UIElement:
        return UIElement(
            text="",
            resource_id=resource_id,
            class_name="android.view.View",
            content_desc="",
            bounds=bounds,
            clickable=False,
            enabled=True,
            checkable=False,
            checked=False,
            focusable=False,
        )

    def test_index_selects_nth_matching_element(self):
        # RecyclerView 里同一 resource-id 会重复出现在每张卡片上（如封面控件），
        # 旧版 find_by_resource_id 永远只返回第一个匹配，无法定位"第二张卡片
        # 的封面"这类场景，导致点击操作只能瞎猜用 class 通配，容易点到无关元素。
        cover1 = self._el_with_id("com.iqiyi.acg:id/iv_card_410_cover", [36, 1090, 356, 1516])
        cover2 = self._el_with_id("com.iqiyi.acg:id/iv_card_410_cover", [380, 1090, 700, 1516])
        screen = ParsedScreen(elements=[cover1, cover2])

        self.assertIs(screen.find_by_resource_id("iv_card_410_cover", index=0), cover1)
        self.assertIs(screen.find_by_resource_id("iv_card_410_cover", index=1), cover2)
        self.assertIsNone(screen.find_by_resource_id("iv_card_410_cover", index=2))

    def test_default_index_matches_original_first_match_behavior(self):
        cover1 = self._el_with_id("com.iqiyi.acg:id/iv_card_410_cover", [36, 1090, 356, 1516])
        screen = ParsedScreen(elements=[cover1])
        self.assertIs(screen.find_by_resource_id("iv_card_410_cover"), cover1)


if __name__ == "__main__":
    unittest.main()
