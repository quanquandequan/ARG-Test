"""DeviceTool：Appium 设备 / 会话管理。

处理 Android 测试会话的生命周期：
  connect        打开新的 Appium 会话（永不自动启动应用）
  disconnect     关闭当前会话
  list_devices   列出已连接的 ADB 设备
  list_packages  列出设备上已安装包（支持关键词过滤）
  launch_app     通过 activate_app 激活 / 启动已安装应用
  install_app    推送并安装 APK

任何移动端自动化场景中都应先调用此工具。
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
        "list_devices（列出ADB设备）、list_packages（查找已安装包名）、"
        "launch_app（启动应用）、install_app（安装APK）。"
        "在执行任何 UI 操作前，必须先调用 connect 建立连接。"
        "不知道包名时，先调用 list_packages 搜索，不要猜包名。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "connect", "disconnect", "list_devices",
                    "list_packages", "launch_app", "install_app",
                ],
                "description": "要执行的操作",
            },
            "server_url": {
                "type": "string",
                "description": "Appium 服务地址，如 http://localhost:4723（action=connect 时使用）",
            },
            "device_name": {
                "type": "string",
                "description": "设备序列号，如 emulator-5554（action=connect 时可选，优先读配置）",
            },
            "platform_version": {
                "type": "string",
                "description": "Android 版本（action=connect 时可选，优先读配置）",
            },
            "keyword": {
                "type": "string",
                "description": "包名关键字过滤（action=list_packages 时使用，如 bada、iqiyi）",
            },
            "app_package": {
                "type": "string",
                "description": "应用包名，如 com.iqiyi.acg（action=launch_app 时使用，可选读配置）",
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

    # ── 入口 ─────────────────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs) -> str:  # type: ignore[override]
        action = action.strip().lower()

        if action == "connect":
            return await self._connect(**kwargs)
        if action == "disconnect":
            return await self._disconnect()
        if action == "list_devices":
            return await self._list_devices()
        if action == "list_packages":
            return await self._list_packages(**kwargs)
        if action == "launch_app":
            return await self._launch_app(**kwargs)
        if action == "install_app":
            return await self._install_app(**kwargs)

        return (
            f"未知操作：{action}。"
            "支持：connect / disconnect / list_devices / launch_app / install_app"
        )

    # ── 操作 ─────────────────────────────────────────────────────────────────

    async def _connect(
        self,
        server_url: str = "",
        device_name: str = "",
        platform_version: str = "",
        **_,
    ) -> str:
        cfg_mobile = get_config().get("mobile", {})

        # 配置值是权威来源；这里忽略 LLM 提供的 app_package / app_activity，
        # 它们只由 launch_app 使用，不参与会话创建。
        server_url = server_url or cfg_mobile.get("appium_server_url", "http://localhost:4723")
        # 优先使用配置中的 device_name/platform_version；配置为空时才回退到参数。
        device_name = cfg_mobile.get("device_name", "") or device_name
        platform_version = str(cfg_mobile.get("platform_version", "")) or platform_version

        caps: dict = {
            "platformName": "Android",
            "appium:automationName": "UIAutomator2",
            "appium:newCommandTimeout": int(cfg_mobile.get("new_command_timeout", 300)),
            # 不要在这里设置 appPackage / appActivity。
            # 设置后 Appium 会通过 `am start-activity` 自动启动应用，
            # 对非 exported Activity 会失败（EMUI 等系统上抛 SecurityException）。
            # 应用启动由 launch_app / activate_app 单独处理。
        }
        if device_name:
            caps["appium:deviceName"] = device_name
        if platform_version:
            caps["appium:platformVersion"] = platform_version

        # EMUI / 华为设备：避免每次弹"安装确认"弹窗
        if cfg_mobile.get("skip_server_installation", False):
            caps["appium:skipServerInstallation"] = True
        if cfg_mobile.get("no_reset", False):
            caps["appium:noReset"] = True
        if cfg_mobile.get("auto_grant_permissions", False):
            caps["appium:autoGrantPermissions"] = True

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
        **_,
    ) -> str:
        if not self._mgr.is_connected():
            return "错误：请先调用 connect 建立设备连接。"

        # LLM 未提供包名时回退到 YAML 配置
        if not app_package:
            cfg_mobile = get_config().get("mobile", {})
            app_package = cfg_mobile.get("app_package", "")
        if not app_package:
            return "错误：必须提供 app_package 参数（或在 mobile.app_package 配置中设置）。"

        # 如果 app 已在前台运行，跳过 activate_app。
        # activate_app 在 app 已处于前台时会触发其启动流程（splash → 导航），
        # 可能导致 app 退到后台或进入错误页面。
        current_pkg = await self._mgr.get_current_package()
        if current_pkg == app_package:
            activity = await self._mgr.get_current_activity()
            return f"应用 {app_package} 已在前台，无需重新启动。当前 Activity：{activity or '未知'}"

        try:
            # launch_app 内部使用 activate_app，不存在 SecurityException 风险
            await self._mgr.launch_app(app_package)
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
