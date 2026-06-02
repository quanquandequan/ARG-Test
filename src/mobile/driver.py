"""Appium driver manager — wraps the Appium WebDriver session.

All blocking Appium operations run in a thread pool via ``asyncio.to_thread``
so they don't block the async event loop.

Usage:
    mgr = AppiumDriverManager()
    await mgr.connect(server_url="http://localhost:4723", caps={...})
    xml = await mgr.get_page_source()
    b64 = await mgr.get_screenshot_base64()
    await mgr.tap(x=540, y=960)
    await mgr.disconnect()

Install dependency:
    pip install appium-python-client Pillow
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger(__name__)


class AppiumDriverManager:
    """Manages a single Appium WebDriver session.

    Thread-safe for use in async context: blocking Appium calls are executed
    via ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self._driver = None  # appium.webdriver.Remote instance

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return self._driver is not None

    async def connect(
        self,
        server_url: str,
        caps: dict,
    ) -> None:
        """Open an Appium session with the given desired capabilities.

        Args:
            server_url: Appium server URL, e.g. "http://localhost:4723"
            caps: Appium desired capabilities dict.
        """
        if self._driver is not None:
            await self.disconnect()

        def _connect():
            try:
                from appium import webdriver
                from appium.options import AppiumOptions
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
        """Quit the current Appium session."""
        if self._driver is None:
            return
        driver = self._driver
        self._driver = None

        def _quit():
            try:
                driver.quit()
            except Exception:
                pass  # already dead

        await asyncio.to_thread(_quit)
        logger.info("appium_disconnected")

    def _require_driver(self):
        if self._driver is None:
            raise RuntimeError(
                "设备未连接，请先调用 device_tool action=connect 建立 Appium 会话。"
            )
        return self._driver

    # ── Screen capture ────────────────────────────────────────────────────────

    async def get_page_source(self) -> str:
        """Return the current XML page source (UIAutomator2 layout hierarchy)."""
        drv = self._require_driver()
        return await asyncio.to_thread(lambda: drv.page_source)

    async def get_screenshot_base64(self) -> str:
        """Capture a screenshot and return it as a base64-encoded PNG string."""
        drv = self._require_driver()
        return await asyncio.to_thread(lambda: drv.get_screenshot_as_base64())

    async def save_screenshot(self, path: str | Path) -> Path:
        """Save screenshot to disk and return the absolute path."""
        b64 = await self.get_screenshot_base64()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(b64))
        logger.debug("screenshot_saved", path=str(dest))
        return dest.resolve()

    async def get_current_activity(self) -> str:
        """Return the current Android activity name."""
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

    # ── App management ────────────────────────────────────────────────────────

    async def launch_app(self, package: str, activity: str = "") -> None:
        """Activate / start an installed app by package name."""
        drv = self._require_driver()

        def _launch():
            if activity:
                drv.start_activity(package, activity)
            else:
                drv.activate_app(package)

        await asyncio.to_thread(_launch)
        logger.info("app_launched", package=package)

    async def install_app(self, apk_path: str) -> None:
        """Install an APK onto the connected device."""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.install_app(apk_path))
        logger.info("app_installed", apk_path=apk_path)

    async def terminate_app(self, package: str) -> None:
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.terminate_app(package))

    # ── Touch actions ─────────────────────────────────────────────────────────

    async def tap(self, x: int, y: int) -> None:
        """Tap at screen coordinates (pixels)."""
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
        """Swipe from (start_x, start_y) to (end_x, end_y)."""
        drv = self._require_driver()
        await asyncio.to_thread(
            lambda: drv.swipe(start_x, start_y, end_x, end_y, duration_ms)
        )
        logger.debug("swipe", start=(start_x, start_y), end=(end_x, end_y))

    # ── Text input ────────────────────────────────────────────────────────────

    async def input_text(self, text: str) -> None:
        """Type text into the currently focused element."""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.execute_script("mobile: type", {"text": text}))
        logger.debug("input_text", length=len(text))

    async def clear_focused_field(self) -> None:
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.execute_script("mobile: clearTextField"))

    # ── System ────────────────────────────────────────────────────────────────

    async def press_back(self) -> None:
        """Press the Android back button."""
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.press_keycode(4))  # 4 = KEYCODE_BACK
        logger.debug("press_back")

    async def press_home(self) -> None:
        drv = self._require_driver()
        await asyncio.to_thread(lambda: drv.press_keycode(3))  # 3 = KEYCODE_HOME

    async def list_devices(self) -> list[str]:
        """List connected Android devices via adb (does not require Appium session)."""

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
                for line in lines[1:]      # skip "List of devices attached"
                if "\tdevice" in line
            ]
            return devices
        except FileNotFoundError:
            return []  # adb not in PATH


# ── Module-level singleton ────────────────────────────────────────────────────
# All mobile tools share this single manager instance.

_MANAGER: AppiumDriverManager | None = None


def get_driver_manager() -> AppiumDriverManager:
    """Return (or create) the module-level AppiumDriverManager singleton."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AppiumDriverManager()
    return _MANAGER
