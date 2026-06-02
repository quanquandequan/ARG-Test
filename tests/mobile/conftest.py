"""Shared fixtures for mobile tool tests.

All tests use FakeAppiumDriverManager — no real Appium / device required.
"""

from __future__ import annotations

import pytest

from src.mobile.driver import AppiumDriverManager
from src.services.page_cache import PageCache

# ── Sample XML fixtures ───────────────────────────────────────────────────────

LOGIN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy>
  <android.widget.FrameLayout bounds="[0,0][1080,2340]" clickable="false">
    <android.widget.TextView
        text="欢迎登录"
        resource-id="com.example:id/title"
        bounds="[280,200][800,270]"
        clickable="false"/>
    <android.widget.EditText
        text=""
        content-desc="用户名输入框"
        resource-id="com.example:id/username"
        bounds="[80,320][1000,400]"
        clickable="true"/>
    <android.widget.EditText
        text=""
        content-desc="密码输入框"
        resource-id="com.example:id/password"
        bounds="[80,420][1000,500]"
        clickable="true"/>
    <android.widget.Button
        text="登录"
        resource-id="com.example:id/login_btn"
        bounds="[200,560][880,630]"
        clickable="true"/>
    <android.widget.TextView
        text="忘记密码？"
        resource-id="com.example:id/forgot"
        bounds="[400,650][680,700]"
        clickable="true"/>
  </android.widget.FrameLayout>
</hierarchy>
"""

HOME_XML = """\
<hierarchy>
  <android.widget.FrameLayout bounds="[0,0][1080,2340]">
    <android.widget.TextView
        text="首页"
        resource-id="com.example:id/home_title"
        bounds="[100,50][980,100]"
        clickable="false"/>
    <android.widget.Button
        text="发现"
        resource-id="com.example:id/discover"
        bounds="[200,1000][880,1080]"
        clickable="true"/>
    <android.widget.CheckBox
        text="记住我"
        resource-id="com.example:id/remember"
        bounds="[80,700][200,750]"
        clickable="true"
        checkable="true"
        checked="true"/>
  </android.widget.FrameLayout>
</hierarchy>
"""


class FakeAppiumDriverManager(AppiumDriverManager):
    """Fake driver manager for testing without a real Appium session."""

    def __init__(
        self,
        page_source: str = LOGIN_XML,
        activity: str = ".LoginActivity",
        package: str = "com.example.app",
        screenshot_b64: str = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ),
        devices: list[str] | None = None,
    ) -> None:
        self._connected = False
        self._page_source = page_source
        self._activity = activity
        self._package = package
        self._screenshot_b64 = screenshot_b64
        self._devices = devices if devices is not None else ["emulator-5554"]
        self._tapped: list[tuple[int, int]] = []
        self._typed: list[str] = []
        self._swiped: list[dict] = []
        self._driver = None  # satisfy parent's _require_driver logic

    def is_connected(self) -> bool:
        return self._connected

    def _require_driver(self):
        if not self._connected:
            raise RuntimeError("设备未连接，请先调用 device_tool action=connect 建立 Appium 会话。")
        return self  # return self as a fake driver

    async def connect(self, server_url: str, caps: dict) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_page_source(self) -> str:
        self._require_driver()
        return self._page_source

    async def get_screenshot_base64(self) -> str:
        self._require_driver()
        return self._screenshot_b64

    async def save_screenshot(self, path) -> Path:  # noqa: F821
        from pathlib import Path
        self._require_driver()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        import base64
        dest.write_bytes(base64.b64decode(self._screenshot_b64))
        return dest.resolve()

    async def get_current_activity(self) -> str:
        return self._activity if self._connected else ""

    async def get_current_package(self) -> str:
        return self._package if self._connected else ""

    async def launch_app(self, package: str, activity: str = "") -> None:
        self._require_driver()

    async def install_app(self, apk_path: str) -> None:
        self._require_driver()

    async def terminate_app(self, package: str) -> None:
        self._require_driver()

    async def tap(self, x: int, y: int) -> None:
        self._require_driver()
        self._tapped.append((x, y))

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        self._require_driver()

    async def swipe(self, sx: int, sy: int, ex: int, ey: int, duration_ms: int = 800) -> None:
        self._require_driver()
        self._swiped.append({"from": (sx, sy), "to": (ex, ey)})

    async def input_text(self, text: str) -> None:
        self._require_driver()
        self._typed.append(text)

    async def clear_focused_field(self) -> None:
        self._require_driver()

    async def press_back(self) -> None:
        self._require_driver()

    async def press_home(self) -> None:
        self._require_driver()

    async def list_devices(self) -> list[str]:
        return self._devices

    def set_page_source(self, xml: str) -> None:
        self._page_source = xml

    def set_activity(self, activity: str) -> None:
        self._activity = activity


# ── pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def fake_driver() -> FakeAppiumDriverManager:
    return FakeAppiumDriverManager()


@pytest.fixture
def connected_driver() -> FakeAppiumDriverManager:
    drv = FakeAppiumDriverManager()
    drv._connected = True
    return drv


@pytest.fixture
def page_cache() -> PageCache:
    return PageCache(ttl_seconds=30.0)
