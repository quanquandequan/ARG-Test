"""Appium driver manager：封装 Appium WebDriver 会话。

所有阻塞型 Appium 操作都通过 ``asyncio.to_thread`` 在线程池中执行，
避免阻塞异步事件循环。

用法：
    mgr = AppiumDriverManager()
    await mgr.connect(server_url="http://localhost:4723", caps={...})
    xml = await mgr.get_page_source()
    b64 = await mgr.get_screenshot_base64()
    await mgr.tap(x=540, y=960)
    await mgr.disconnect()

安装依赖：
    pip install appium-python-client Pillow
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger(__name__)


class AppiumDriverManager:
    """管理单个 Appium WebDriver 会话。

    可在异步上下文中安全使用：阻塞型 Appium 调用通过
    ``asyncio.to_thread`` 执行。
    """

    def __init__(self) -> None:
        self._driver = None  # appium.webdriver.Remote 实例

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return self._driver is not None

    async def probe_session_alive(self) -> bool:
        """探测已持有的 Appium 会话在服务端是否仍然存活。

        ``is_connected()`` 只判断本地是否持有 driver 引用，不代表服务端会话
        没有过期——例如 ``newCommandTimeout``（默认 300 秒）到期、
        Appium server 被重启或手动杀掉等场景下，本地引用仍然非空，
        但服务端早已销毁会话，后续任何操作都会抛出
        ``NoSuchDriverError: A session is either terminated or not started``。
        这里用一次真实的网络往返（``get_window_size``，各平台都支持）主动探活；
        探测到会话已死时清空本地引用，方便调用方（如 ``ExecutionWorkflow``）
        判断需要重新 connect，而不是直接把陈旧的错误抛给用户。
        """
        if self._driver is None:
            return False
        driver = self._driver

        def _probe():
            driver.get_window_size()

        try:
            await asyncio.to_thread(_probe)
            return True
        except Exception:
            logger.warning("appium_session_stale_detected")
            self._driver = None
            return False

    async def connect(
        self,
        server_url: str,
        caps: dict,
    ) -> None:
        """使用给定 desired capabilities 打开 Appium 会话。

        Args:
            server_url: Appium 服务地址，例如 "http://localhost:4723"。
            caps: Appium desired capabilities 字典。
        """
        if self._driver is not None:
            await self.disconnect()

        def _connect():
            try:
                from appium import webdriver
                from appium.options.common.base import AppiumOptions
            except ImportError as e:
                raise RuntimeError(
                    "appium-python-client not installed. "
                    "Run: pip install appium-python-client"
                ) from e

            options = AppiumOptions()
            options.load_capabilities(caps)
            return webdriver.Remote(server_url, options=options)

        self._driver = await asyncio.to_thread(_connect)
        logger.info("appium_connected", server_url=server_url)

    async def disconnect(self) -> None:
        """退出当前 Appium 会话。"""
        if self._driver is None:
            return
        driver = self._driver
        self._driver = None

        def _quit():
            try:
                driver.quit()
            except Exception:
                pass  # 会话已经失效

        await asyncio.to_thread(_quit)
        logger.info("appium_disconnected")

    def _require_driver(self):
        if self._driver is None:
            raise RuntimeError(
                "设备未连接，请先调用 device_tool action=connect 建立 Appium 会话。"
            )
        return self._driver

    # ── 屏幕采集 ─────────────────────────────────────────────────────────────

    async def get_page_source(self) -> str:
        """返回当前 XML 页面源码（UIAutomator2 布局层级）。"""
        drv = self._require_driver()
        return await asyncio.to_thread(lambda: drv.page_source)

    async def get_parsed_screen(self):
        """一次调用获取并解析当前屏幕。

        对 ``get_page_source()`` + ``parse_page_source()`` 的便捷封装。
        """
        from src.mobile.screen_parser import parse_page_source

        xml = await self.get_page_source()
        return parse_page_source(xml)

    async def get_screenshot_base64(self) -> str:
        """截图并返回 base64 编码的 PNG 字符串。"""
        drv = self._require_driver()
        return await asyncio.to_thread(lambda: drv.get_screenshot_as_base64())

    async def save_screenshot(self, path: str | Path) -> Path:
        """将截图保存到磁盘并返回绝对路径。"""
        b64 = await self.get_screenshot_base64()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(b64))
        logger.debug("screenshot_saved", path=str(dest))
        return dest.resolve()

    async def get_current_activity(self) -> str:
        """返回当前 Android activity 名称。"""
        drv = self._require_driver()
        try:
            return await asyncio.to_thread(lambda: drv.current_activity or "")
        except Exception:
            return ""

    async def get_current_package(self) -> str:
        drv = self._require_driver()
        try:
            return await asyncio.to_thread(lambda: drv.current_package or "")
        except Exception:
            return ""

    # ── App 管理 ─────────────────────────────────────────────────────────────

    async def launch_app(self, package: str) -> None:
        """按包名激活已安装应用。

        使用 ``activate_app``（等价于点击启动器图标）。
        这里刻意避免 ``start_activity``，因为 Android 会拒绝对非 exported
        Activity 的显式 intent 启动（EMUI 和其他 OEM 上会触发 SecurityException）。
        """
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.activate_app(package))
        logger.info("app_launched", package=package)

    async def install_app(self, apk_path: str) -> None:
        """将 APK 安装到已连接设备。"""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.install_app(apk_path))
        logger.info("app_installed", apk_path=apk_path)

    async def terminate_app(self, package: str) -> None:
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.terminate_app(package))

    # ── 触控操作 ─────────────────────────────────────────────────────────────

    async def tap(self, x: int, y: int) -> None:
        """按屏幕坐标点击（像素）。"""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.tap([(x, y)]))
        logger.debug("tap", x=x, y=y)

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        drv = self._require_driver()

        def _long_press():
            from appium.webdriver.common.touch_action import TouchAction
            TouchAction(drv).long_press(x=x, y=y, duration=duration_ms).perform()

        await asyncio.to_thread(_long_press)

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 800,
    ) -> None:
        """从 (start_x, start_y) 滑动到 (end_x, end_y)。"""
        drv = self._require_driver()
        await asyncio.to_thread(
            lambda: drv.swipe(start_x, start_y, end_x, end_y, duration_ms)
        )
        logger.debug("swipe", start=(start_x, start_y), end=(end_x, end_y))

    # ── 文本输入 ─────────────────────────────────────────────────────────────

    async def input_text(self, text: str) -> None:
        """向当前聚焦元素输入文本。"""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.execute_script("mobile: type", {"text": text}))
        logger.debug("input_text", length=len(text))

    async def clear_focused_field(self) -> None:
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.execute_script("mobile: clearTextField"))

    # ── 系统操作 ─────────────────────────────────────────────────────────────

    async def press_back(self) -> None:
        """按下 Android 返回键。"""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.press_keycode(4))  # 4 = KEYCODE_BACK
        logger.debug("press_back")

    async def press_home(self) -> None:
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.press_keycode(3))  # 3 = KEYCODE_HOME

    async def list_devices(self) -> list[str]:
        """通过 adb 列出已连接 Android 设备（无需 Appium 会话）。"""

        try:
            proc = await asyncio.create_subprocess_exec(
                "adb", "devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode().strip().splitlines()
            devices = [
                line.split("\t")[0]
                for line in lines[1:]      # 跳过 "List of devices attached"
                if "\tdevice" in line
            ]
            return devices
        except FileNotFoundError:
            return []  # adb 不在 PATH 中


# ── 模块级单例 ───────────────────────────────────────────────────────────────
# 所有移动端工具共享这个 manager 实例。

_MANAGER: AppiumDriverManager | None = None


def get_driver_manager() -> AppiumDriverManager:
    """返回（或创建）模块级 AppiumDriverManager 单例。"""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AppiumDriverManager()
    return _MANAGER
