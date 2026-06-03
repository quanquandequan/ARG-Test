"""ScreenTool — hybrid screen analysis (XML tree primary, Qwen VL fallback).

Strategy (XML-first):
  1. Fetch page_source XML from Appium.
  2. Check PageCache — return cached result if page structure is unchanged.
  3. Parse XML into a structured element list.
  4. Decide if XML is "sufficient":
       - ANY clickable/focusable element exists  → XML is sufficient
       - OR text/content-desc count >= threshold → XML is sufficient
       (threshold configurable via mobile.vlm_fallback_min_text_elements in YAML,
        default 1 — VLM fires only when XML is truly empty)
  5. If sufficient  → return XML result directly (no screenshot taken).
  6. If insufficient (or force_vlm=true) → take screenshot, call Qwen VL,
     return VLM description alongside whatever XML elements were found.
  7. Cache the result.

Supported actions:
  get_current_screen   XML-first hybrid analysis of the current screen.
  get_screenshot       Take and save a screenshot (returns file path).
  get_ui_tree          Return the raw XML element list (never calls VLM).
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

    # ── Entry point ───────────────────────────────────────────────────────────

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

    # ── Actions ───────────────────────────────────────────────────────────────

    async def _get_current_screen(self, force_vlm: bool = False) -> str:
        """Hybrid screen analysis — XML primary, VLM fallback."""
        if not self._mgr.is_connected():
            return "错误：设备未连接，请先调用 device_tool action=connect。"

        try:
            xml = await self._mgr.get_page_source()
        except Exception as e:
            err = str(e)
            if "terminated" in err or "not started" in err or "session" in err.lower():
                await self._mgr.disconnect()  # calls quit() + sets _driver=None cleanly
                return (
                    "Appium 会话已失效（超时或服务重启）。"
                    "请调用 device_tool action=connect 重新建立连接，再继续操作。"
                )
            return f"获取页面失败：{e}"

        page_hash = compute_structure_hash(xml)
        activity = await self._mgr.get_current_activity()

        # Cache hit
        cached = self._cache.get(page_hash)
        if cached and not force_vlm:
            logger.debug("screen_cache_hit", page_hash=page_hash)
            return _format_screen_result(
                cached.parsed_screen,
                activity=cached.activity,
                source="cache",
                vlm_description=None,
            )

        # Parse XML
        parsed = parse_page_source(xml)

        vlm_description: str | None = None
        source = "xml"

        # ── VLM fallback decision (XML-first) ─────────────────────────────────
        # Read threshold from YAML; 0 means "always trust XML, never auto-fire".
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

        # Cache the result (without screenshot for memory efficiency)
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


# ── Formatting ────────────────────────────────────────────────────────────────

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
