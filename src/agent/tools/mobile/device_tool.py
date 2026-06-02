"""DeviceTool — Appium device / session management.

Handles the lifecycle of an Android test session:
  connect      Open a new Appium session
  disconnect   Close the current session
  list_devices List connected ADB devices
  launch_app   Activate / start an installed app
  install_app  Push and install an APK

This tool is always called first in any mobile automation scenario.
"""

from __future__ import annotations

from src.agent.base_tool import BaseTool
from src.core.config import get_config
from src.core.logging import get_logger
from src.mobile.driver import AppiumDriverManager

logger = get_logger(__name__)


class DeviceTool(BaseTool):
    name = "device_tool"
    description = (
        "管理 Android 设备的 Appium 测试会话。"
        "支持操作：connect（连接设备）、disconnect（断开）、"
        "list_devices（列出设备）、launch_app（启动应用）、install_app（安装APK）。"
        "在执行任何 UI 操作前，必须先调用 connect 建立连接。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["connect", "disconnect", "list_devices", "launch_app", "install_app"],
                "description": "要执行的操作",
            },
            "server_url": {
                "type": "string",
                "description": "Appium 服务地址，如 http://localhost:4723（action=connect 时必填）",
            },
            "device_name": {
                "type": "string",
                "description": "设备名称或序列号，如 emulator-5554",
            },
            "platform_version": {
                "type": "string",
                "description": "Android 版本，如 14",
            },
            "app_package": {
                "type": "string",
                "description": "应用包名，如 com.example.app",
            },
            "app_activity": {
                "type": "string",
                "description": "启动 Activity，如 .MainActivity",
            },
            "apk_path": {
                "type": "string",
                "description": "APK 文件本地路径（action=install_app 时使用）",
            },
        },
        "required": ["action"],
    }

    def __init__(self, driver_manager: AppiumDriverManager) -> None:
        self._mgr = driver_manager

    # ── Entry point ───────────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs) -> str:  # type: ignore[override]
        action = action.strip().lower()

        if action == "connect":
            return await self._connect(**kwargs)
        if action == "disconnect":
            return await self._disconnect()
        if action == "list_devices":
            return await self._list_devices()
        if action == "launch_app":
            return await self._launch_app(**kwargs)
        if action == "install_app":
            return await self._install_app(**kwargs)

        return (
            f"未知操作：{action}。"
            "支持：connect / disconnect / list_devices / launch_app / install_app"
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    async def _connect(
        self,
        server_url: str = "",
        device_name: str = "",
        platform_version: str = "",
        app_package: str = "",
        app_activity: str = "",
        **_,
    ) -> str:
        cfg_mobile = get_config().get("mobile", {})

        server_url = server_url or cfg_mobile.get("appium_server_url", "http://localhost:4723")
        device_name = device_name or cfg_mobile.get("device_name", "")
        platform_version = platform_version or str(cfg_mobile.get("platform_version", ""))
        app_package = app_package or cfg_mobile.get("app_package", "")
        app_activity = app_activity or cfg_mobile.get("app_activity", "")

        caps: dict = {
            "platformName": "Android",
            "appium:automationName": "UIAutomator2",
            "appium:newCommandTimeout": int(cfg_mobile.get("new_command_timeout", 300)),
        }
        if device_name:
            caps["appium:deviceName"] = device_name
        if platform_version:
            caps["appium:platformVersion"] = platform_version
        if app_package:
            caps["appium:appPackage"] = app_package
        if app_activity:
            caps["appium:appActivity"] = app_activity

        try:
            await self._mgr.connect(server_url=server_url, caps=caps)
        except Exception as e:
            return f"连接失败：{e}"

        activity = await self._mgr.get_current_activity()
        return (
            f"已成功连接到 Appium 服务 {server_url}。\n"
            f"当前 Activity：{activity or '未知'}"
        )

    async def _disconnect(self) -> str:
        if not self._mgr.is_connected():
            return "当前没有活跃的设备连接。"
        await self._mgr.disconnect()
        return "Appium 会话已断开。"

    async def _list_devices(self) -> str:
        devices = await self._mgr.list_devices()
        if not devices:
            return "未发现已连接的 Android 设备（请确认 adb 已安装且设备已授权）。"
        return "已连接设备：\n" + "\n".join(f"  - {d}" for d in devices)

    async def _launch_app(
        self,
        app_package: str = "",
        app_activity: str = "",
        **_,
    ) -> str:
        if not self._mgr.is_connected():
            return "错误：请先调用 connect 建立设备连接。"
        if not app_package:
            return "错误：必须提供 app_package 参数。"
        try:
            await self._mgr.launch_app(app_package, app_activity)
            activity = await self._mgr.get_current_activity()
            return f"应用 {app_package} 已启动，当前 Activity：{activity or '未知'}"
        except Exception as e:
            return f"启动失败：{e}"

    async def _install_app(self, apk_path: str = "", **_) -> str:
        if not self._mgr.is_connected():
            return "错误：请先调用 connect 建立设备连接。"
        if not apk_path:
            return "错误：必须提供 apk_path 参数。"
        try:
            await self._mgr.install_app(apk_path)
            return f"APK 安装成功：{apk_path}"
        except Exception as e:
            return f"安装失败：{e}"
