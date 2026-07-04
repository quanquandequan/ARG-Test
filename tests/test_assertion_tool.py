"""AssertionTool 的 assert_page / assert_not_page 单元测试。

背景：``assert_page`` 只能验证"当前 Activity 匹配指定关键字"，但很多
"点击封面/进入详情"场景的落地页可能是覆盖在同一 Activity 上的半屏浮层，
或是一个自动化用例生成时根本不知道类名的新 Activity——用中文业务名称
（如"播放详情页"）去匹配英文 Activity 类名本来就不可能命中。``assert_not_page``
提供了一种不需要知道目标页面确切标识、只验证"已经离开原页面"的更可靠断言。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.agent.tools.mobile.assertion_tool import AssertionTool


def _make_tool(activity: str, package: str = "com.iqiyi.acg") -> AssertionTool:
    mgr = MagicMock()
    mgr.is_connected.return_value = True
    mgr.get_current_activity = AsyncMock(return_value=activity)
    mgr.get_current_package = AsyncMock(return_value=package)
    return AssertionTool(driver_manager=mgr)


class TestAssertPage(unittest.IsolatedAsyncioTestCase):
    async def test_assert_page_passes_on_keyword_match(self):
        tool = _make_tool(activity=".ChasePageActivity")
        result = await tool.execute(action="assert_page", page="ChasePageActivity")
        self.assertIn("✅ PASS", result)

    async def test_assert_page_fails_when_chinese_keyword_cannot_match_activity(self):
        # 中文业务名称不可能作为子串出现在英文 Activity 类名里，
        # 这类断言几乎注定失败——应改用更贴合实际的关键字或 assert_not_page。
        tool = _make_tool(activity=".biz.cartoon.player.PlayerActivity")
        result = await tool.execute(action="assert_page", page="播放详情页")
        self.assertIn("❌ FAIL", result)


class TestAssertNotPage(unittest.IsolatedAsyncioTestCase):
    async def test_assert_not_page_passes_when_activity_changed(self):
        # 点击封面后跳转到了未知的详情 Activity，只需确认已经离开推荐页。
        tool = _make_tool(activity=".biz.cartoon.player.PlayerActivity")
        result = await tool.execute(
            action="assert_not_page", page="ComicsMainActivity"
        )
        self.assertIn("✅ PASS", result)
        self.assertIn("已离开页面", result)

    async def test_assert_not_page_fails_when_still_on_same_page(self):
        # 复现真实报错场景：点击目标定位错误，实际没有发生任何跳转，
        # 仍停留在原页面——assert_not_page 应该如实报告失败。
        tool = _make_tool(activity=".biz.cartoon.main.ComicsMainActivity")
        result = await tool.execute(
            action="assert_not_page", page="ComicsMainActivity"
        )
        self.assertIn("❌ FAIL", result)
        self.assertIn("仍停留在页面", result)


if __name__ == "__main__":
    unittest.main()
