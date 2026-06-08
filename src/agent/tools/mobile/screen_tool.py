"""ScreenTool：混合屏幕分析（优先 XML 树，Qwen VL 兜底）。

策略（XML 优先）：
  1. 从 Appium 获取 page_source XML。
  2. 检查 PageCache；若页面结构未变，则返回缓存结果。
  3. 将 XML 解析为结构化元素列表。
  4. 判断 XML 是否“足够”：
       - 存在任意 clickable/focusable 元素，则 XML 足够；
       - 或 text/content-desc 数量达到阈值，则 XML 足够。
       阈值可通过 YAML 中的 mobile.vlm_fallback_min_text_elements 配置，
       默认 1，即只有 XML 真的很空时才触发 VLM。
  5. 若足够，则直接返回 XML 结果（不截图）。
  6. 若不足（或 force_vlm=true），则截图并调用 Qwen VL，
     将 VLM 描述与已找到的 XML 元素一并返回。
  7. 缓存结果。

支持的操作：
  get_current_screen   对当前屏幕执行 XML 优先的混合分析。
  get_screenshot       截图并保存（返回文件路径）。
  get_ui_tree          返回原始 XML 元素列表（绝不调用 VLM）。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.core.config import get_config
from src.core.logging import get_logger
from src.mobile.driver import AppiumDriverManager
from src.mobile.screen_parser import ParsedScreen, compute_structure_hash, parse_page_source
from src.services.page_cache import PageCache

logger = get_logger(__name__)


class ScreenTool(BaseTool):
    name = "screen_tool"
    description = (
        "获取当前 Android 屏幕的 UI 信息。"
        "action=get_current_screen 返回结构化的页面元素（优先使用 XML 树，"
        "信息不足时自动调用 VLM 截图识别）。"
        "action=get_screenshot 保存截图到本地。"
        "action=get_ui_tree 返回原始 XML 元素列表（不调用 VLM）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_current_screen", "get_screenshot", "get_ui_tree"],
                "description": "要执行的操作",
            },
            "save_path": {
                "type": "string",
                "description": "截图保存路径（action=get_screenshot 时使用；为空则自动命名）",
            },
            "force_vlm": {
                "type": "boolean",
                "description": "为 true 时强制使用 VLM 分析（即使 XML 已足够）",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        driver_manager: AppiumDriverManager,
        page_cache: PageCache,
        vlm=None,  # QwenVisionProvider | None
    ) -> None:
        self._mgr = driver_manager
        self._cache = page_cache
        self._vlm = vlm

    # ── 入口 ─────────────────────────────────────────────────────────────────

    async def execute(  # type: ignore[override]
        self,
        action: str = "get_current_screen",
        save_path: str = "",
        force_vlm: bool = False,
        **_,
    ) -> str:
        action = action.strip().lower()

        if action == "get_current_screen":
            return await self._get_current_screen(force_vlm=force_vlm)
        if action == "get_screenshot":
            return await self._get_screenshot(save_path=save_path)
        if action == "get_ui_tree":
            return await self._get_ui_tree()

        return f"未知操作：{action}。支持：get_current_screen / get_screenshot / get_ui_tree"

    # ── 操作 ─────────────────────────────────────────────────────────────────

    async def _get_current_screen(self, force_vlm: bool = False) -> str:
        """混合屏幕分析：优先 XML，VLM 兜底。"""
        if not self._mgr.is_connected():
            return "错误：设备未连接，请先调用 device_tool action=connect。"

        try:
            xml = await self._mgr.get_page_source()
        except Exception as e:
            err = str(e)
            if "terminated" in err or "not started" in err or "session" in err.lower():
                await self._mgr.disconnect()  # 调用 quit() 并干净地设置 _driver=None
                return (
                    "Appium 会话已失效（超时或服务重启）。"
                    "请调用 device_tool action=connect 重新建立连接，再继续操作。"
                )
            return f"获取页面失败：{e}"

        page_hash = compute_structure_hash(xml)
        activity = await self._mgr.get_current_activity()

        # 命中缓存
        cached = self._cache.get(page_hash)
        if cached and not force_vlm:
            logger.debug("screen_cache_hit", page_hash=page_hash)
            return _format_screen_result(
                cached.parsed_screen,
                activity=cached.activity,
                source="cache",
                vlm_description=None,
            )

        # 解析 XML
        parsed = parse_page_source(xml)

        vlm_description: str | None = None
        source = "xml"

        # ── VLM 兜底决策（XML 优先）──────────────────────────────────────────
        # 从 YAML 读取阈值；0 表示“始终信任 XML，绝不自动触发”。
        cfg_mobile = get_config().get("mobile", {})
        min_text = int(cfg_mobile.get("vlm_fallback_min_text_elements", 1))
        xml_sufficient = parsed.has_meaningful_content(min_text_elements=min_text)

        use_vlm = (
            (force_vlm or not xml_sufficient)
            and self._vlm is not None
            and self._vlm.is_available()
        )
        if use_vlm:
            reason = "forced" if force_vlm else "xml_insufficient"
            logger.debug("screen_vlm_fallback", reason=reason,
                         xml_elements=len(parsed.elements))
            try:
                screenshot_b64 = await self._mgr.get_screenshot_base64()
                vlm_description = await self._vlm.describe_screen(screenshot_b64)
                source = "vlm" if not xml_sufficient else "xml+vlm"
            except Exception as exc:
                logger.warning("vlm_fallback_failed", error=str(exc))
        else:
            logger.debug(
                "screen_xml_used",
                elements=len(parsed.elements),
                clickable=len(parsed.clickable_elements()),
            )

        # 缓存结果（不保存截图，节省内存）
        self._cache.put(
            page_hash=page_hash,
            parsed_screen=parsed,
            activity=activity,
            screenshot_base64=None,
        )

        return _format_screen_result(
            parsed,
            activity=activity,
            source=source,
            vlm_description=vlm_description,
        )

    async def _get_screenshot(self, save_path: str = "") -> str:
        if not self._mgr.is_connected():
            return "错误：设备未连接，请先调用 device_tool action=connect。"

        cfg = get_config().get("mobile", {})
        if not save_path:
            import time
            ts = int(time.time())
            screenshot_dir = cfg.get("screenshot_dir", "./outputs/screenshots")
            save_path = str(Path(screenshot_dir) / f"screenshot_{ts}.png")

        try:
            saved = await self._mgr.save_screenshot(save_path)
            return f"截图已保存：{saved}"
        except Exception as e:
            return f"截图保存失败：{e}"

    async def _get_ui_tree(self) -> str:
        if not self._mgr.is_connected():
            return "错误：设备未连接，请先调用 device_tool action=connect。"

        xml = await self._mgr.get_page_source()
        parsed = parse_page_source(xml)
        activity = await self._mgr.get_current_activity()

        summary = parsed.to_agent_summary()
        return (
            f"当前 Activity：{activity}\n"
            f"页面元素（共 {summary['element_count']} 个可见元素，"
            f"{summary['clickable_count']} 个可点击）：\n"
            + json.dumps(summary["elements"], ensure_ascii=False, indent=2)
        )


# ── 格式化 ───────────────────────────────────────────────────────────────────

def _format_screen_result(
    parsed: ParsedScreen,
    activity: str,
    source: str,
    vlm_description: str | None,
) -> str:
    summary = parsed.to_agent_summary()
    lines = [f"当前 Activity：{activity}", f"数据来源：{source}"]

    if vlm_description:
        lines.append("\n【VLM 页面描述】")
        lines.append(vlm_description.strip())
        lines.append("")

    lines.append(
        f"【XML 元素列表】共 {summary['element_count']} 个可见元素，"
        f"{summary['clickable_count']} 个可点击："
    )
    lines.append(json.dumps(summary["elements"], ensure_ascii=False, indent=2))

    return "\n".join(lines)
