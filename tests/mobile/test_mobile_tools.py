"""Tests for Device, Screen, Action, and Assertion tools.

All tests use FakeAppiumDriverManager — no real Appium / device required.
"""

from __future__ import annotations

import pytest

from src.agent.tools.mobile.action_tool import ActionTool
from src.agent.tools.mobile.assertion_tool import AssertionTool
from src.agent.tools.mobile.device_tool import DeviceTool
from src.agent.tools.mobile.screen_tool import ScreenTool
from src.mobile.screen_parser import compute_structure_hash, parse_page_source

from .conftest import HOME_XML, LOGIN_XML, FakeAppiumDriverManager

# ── DeviceTool ────────────────────────────────────────────────────────────────

class TestDeviceTool:
    __test__ = True

    @pytest.fixture
    def tool(self, fake_driver):
        return DeviceTool(driver_manager=fake_driver)

    @pytest.mark.asyncio
    async def test_connect_succeeds(self, tool, fake_driver):
        result = await tool.execute(action="connect")
        assert fake_driver.is_connected()
        assert "成功连接" in result

    @pytest.mark.asyncio
    async def test_connect_returns_activity(self, tool):
        result = await tool.execute(action="connect")
        assert "LoginActivity" in result

    @pytest.mark.asyncio
    async def test_disconnect_when_connected(self, tool, fake_driver):
        fake_driver._connected = True
        result = await tool.execute(action="disconnect")
        assert not fake_driver.is_connected()
        assert "断开" in result

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, tool, fake_driver):
        result = await tool.execute(action="disconnect")
        assert "没有活跃" in result

    @pytest.mark.asyncio
    async def test_list_devices(self, tool):
        result = await tool.execute(action="list_devices")
        assert "emulator-5554" in result

    @pytest.mark.asyncio
    async def test_list_devices_empty(self):
        drv = FakeAppiumDriverManager(devices=[])
        tool = DeviceTool(driver_manager=drv)
        result = await tool.execute(action="list_devices")
        assert "未发现" in result

    @pytest.mark.asyncio
    async def test_launch_app_requires_connection(self, tool, fake_driver):
        result = await tool.execute(action="launch_app", app_package="com.example")
        assert "请先调用 connect" in result

    @pytest.mark.asyncio
    async def test_launch_app_succeeds(self, tool, fake_driver):
        fake_driver._connected = True
        result = await tool.execute(action="launch_app", app_package="com.example.app")
        assert "已启动" in result or "com.example.app" in result

    @pytest.mark.asyncio
    async def test_launch_app_missing_package(self, tool, fake_driver):
        fake_driver._connected = True
        result = await tool.execute(action="launch_app")
        assert "app_package" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="fly")
        assert "未知操作" in result

    def test_tool_schema_has_required_fields(self, tool):
        schema = tool.to_tool_schema()
        assert schema["name"] == "device_tool"
        assert "action" in schema["parameters"]["properties"]


# ── ScreenTool ────────────────────────────────────────────────────────────────

class TestScreenTool:
    __test__ = True

    @pytest.fixture
    def tool(self, connected_driver, page_cache):
        return ScreenTool(
            driver_manager=connected_driver,
            page_cache=page_cache,
            vlm=None,
        )

    @pytest.mark.asyncio
    async def test_get_current_screen_returns_elements(self, tool):
        result = await tool.execute(action="get_current_screen")
        assert "登录" in result or "LoginActivity" in result

    @pytest.mark.asyncio
    async def test_get_current_screen_not_connected(self, fake_driver, page_cache):
        tool = ScreenTool(driver_manager=fake_driver, page_cache=page_cache)
        result = await tool.execute(action="get_current_screen")
        assert "未连接" in result

    @pytest.mark.asyncio
    async def test_get_current_screen_caches_result(self, tool, connected_driver, page_cache):
        await tool.execute(action="get_current_screen")
        hash_key = compute_structure_hash(LOGIN_XML)
        assert page_cache.get(hash_key) is not None

    @pytest.mark.asyncio
    async def test_second_call_hits_cache(self, tool, connected_driver, page_cache):
        # First call populates cache
        await tool.execute(action="get_current_screen")

        # Corrupt the driver's page source to detect if driver is called again
        connected_driver._page_source = "<hierarchy/>"  # would return different content
        result = await tool.execute(action="get_current_screen")
        # Cache hit: result should still contain original login page content
        assert "LoginActivity" in result

    @pytest.mark.asyncio
    async def test_get_ui_tree(self, tool):
        result = await tool.execute(action="get_ui_tree")
        assert "元素" in result
        assert "LoginActivity" in result

    @pytest.mark.asyncio
    async def test_get_screenshot(self, tool, tmp_path):
        save_path = str(tmp_path / "test.png")
        result = await tool.execute(action="get_screenshot", save_path=save_path)
        assert "截图已保存" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="unknown")
        assert "未知操作" in result

    def test_tool_schema(self, tool):
        schema = tool.to_tool_schema()
        assert schema["name"] == "screen_tool"

    @pytest.mark.asyncio
    async def test_vlm_not_called_when_xml_rich(self, connected_driver, page_cache):
        """When XML has enough elements, VLM should not be called."""
        call_count = [0]

        class FakeVLM:
            def is_available(self):
                return True

            async def describe_screen(self, screenshot, prompt=None):
                call_count[0] += 1
                return '{"page_name": "登录页", "elements": []}'

        tool = ScreenTool(driver_manager=connected_driver, page_cache=page_cache, vlm=FakeVLM())
        await tool.execute(action="get_current_screen")
        assert call_count[0] == 0  # VLM not called — XML was sufficient


# ── ActionTool ────────────────────────────────────────────────────────────────

class TestActionTool:
    __test__ = True

    @pytest.fixture
    def tool(self, connected_driver, page_cache):
        return ActionTool(driver_manager=connected_driver, page_cache=page_cache)

    @pytest.mark.asyncio
    async def test_tap_by_text(self, tool, connected_driver):
        # Use unique text that doesn't match "欢迎登录" via fuzzy match
        result = await tool.execute(action="tap", target="忘记密码？")
        assert "已点击" in result
        assert len(connected_driver._tapped) == 1
        cx, cy = connected_driver._tapped[0]
        # "忘记密码？" center: (400+680)//2=540, (650+700)//2=675
        assert cx == 540
        assert cy == 675

    @pytest.mark.asyncio
    async def test_tap_by_coords(self, tool, connected_driver):
        result = await tool.execute(action="tap", x=100, y=200)
        assert "已点击" in result
        assert (100, 200) in connected_driver._tapped

    @pytest.mark.asyncio
    async def test_tap_nonexistent_text(self, tool):
        result = await tool.execute(action="tap", target="不存在的按钮")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_tap_invalidates_cache(self, tool, page_cache):
        # Pre-populate cache
        screen = parse_page_source(LOGIN_XML)
        page_cache.put("h1", screen)
        assert page_cache.size() == 1

        await tool.execute(action="tap", x=100, y=200)
        assert page_cache.size() == 0  # cache invalidated

    @pytest.mark.asyncio
    async def test_input_text(self, tool, connected_driver):
        result = await tool.execute(action="input_text", text="hello@example.com")
        assert "已输入" in result
        assert "hello@example.com" in connected_driver._typed

    @pytest.mark.asyncio
    async def test_input_text_missing_text_param(self, tool):
        result = await tool.execute(action="input_text")
        assert "必须提供" in result

    @pytest.mark.asyncio
    async def test_swipe_up(self, tool, connected_driver):
        result = await tool.execute(action="swipe", direction="up")
        assert "向 up 滑动" in result
        assert len(connected_driver._swiped) == 1

    @pytest.mark.asyncio
    async def test_swipe_unknown_direction(self, tool):
        result = await tool.execute(action="swipe", direction="diagonal")
        assert "未知滑动方向" in result

    @pytest.mark.asyncio
    async def test_back(self, tool):
        result = await tool.execute(action="back")
        assert "返回键" in result

    @pytest.mark.asyncio
    async def test_not_connected_returns_error(self, fake_driver, page_cache):
        tool = ActionTool(driver_manager=fake_driver, page_cache=page_cache)
        result = await tool.execute(action="tap", x=100, y=200)
        assert "未连接" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="fly")
        assert "未知操作" in result

    def test_tool_schema(self, tool):
        schema = tool.to_tool_schema()
        assert schema["name"] == "action_tool"
        assert "action" in schema["parameters"]["required"]


# ── AssertionTool ─────────────────────────────────────────────────────────────

class TestAssertionTool:
    __test__ = True

    @pytest.fixture
    def tool(self, connected_driver):
        return AssertionTool(driver_manager=connected_driver)

    @pytest.mark.asyncio
    async def test_assert_text_passes(self, tool):
        result = await tool.execute(action="assert_text", text="登录")
        assert "PASS" in result
        assert "登录" in result

    @pytest.mark.asyncio
    async def test_assert_text_fails(self, tool):
        result = await tool.execute(action="assert_text", text="不存在")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_assert_not_text_passes(self, tool):
        result = await tool.execute(action="assert_not_text", text="注销")
        assert "PASS" in result

    @pytest.mark.asyncio
    async def test_assert_not_text_fails(self, tool):
        result = await tool.execute(action="assert_not_text", text="登录")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_assert_element_by_id(self, tool):
        result = await tool.execute(action="assert_element", element_id="login_btn")
        assert "PASS" in result

    @pytest.mark.asyncio
    async def test_assert_element_not_found(self, tool):
        result = await tool.execute(action="assert_element", element_id="no_such_btn")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_assert_clickable_passes(self, tool):
        # "忘记密码？" is a clickable TextView (avoids "欢迎登录" fuzzy match)
        result = await tool.execute(action="assert_clickable", element_text="忘记密码？")
        assert "PASS" in result

    @pytest.mark.asyncio
    async def test_assert_clickable_fails_for_non_clickable(self, tool):
        # "欢迎登录" is a TextView with clickable=false
        result = await tool.execute(action="assert_clickable", element_text="欢迎登录")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_assert_page_passes(self, tool):
        result = await tool.execute(action="assert_page", page="LoginActivity")
        assert "PASS" in result

    @pytest.mark.asyncio
    async def test_assert_page_fails(self, tool):
        result = await tool.execute(action="assert_page", page="HomeActivity")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_assert_checked_passes(self, connected_driver):
        """Test checked state using HOME_XML which has a checked CheckBox."""
        connected_driver.set_page_source(HOME_XML)
        tool = AssertionTool(driver_manager=connected_driver)
        result = await tool.execute(action="assert_checked", element_text="记住我")
        assert "PASS" in result

    @pytest.mark.asyncio
    async def test_assert_not_connected(self, fake_driver):
        tool = AssertionTool(driver_manager=fake_driver)
        result = await tool.execute(action="assert_text", text="anything")
        assert "FAIL" in result
        assert "未连接" in result

    @pytest.mark.asyncio
    async def test_missing_text_param(self, tool):
        result = await tool.execute(action="assert_text")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_missing_page_param(self, tool):
        result = await tool.execute(action="assert_page")
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="unknown_assert")
        assert "未知断言" in result

    def test_tool_schema(self, tool):
        schema = tool.to_tool_schema()
        assert schema["name"] == "assertion_tool"
