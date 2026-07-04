"""将 Appium page_source XML 解析为结构化元素列表。

Appium Android（UIAutomator2）返回的 XML 类似：
  <hierarchy>
    <android.widget.FrameLayout bounds="[0,0][1080,2340]">
      <android.widget.Button text="登录" resource-id="com.pkg:id/login_btn"
        bounds="[200,500][880,570]" clickable="true"/>
    </android.widget.FrameLayout>
  </hierarchy>

这里只提取对测试 Agent 有用的元素：
  - 包含 text、content-desc 或已知 resource-id
  - 可见（bounds 未折叠为零面积矩形）
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class UIElement:
    """从页面源码中提取的单个 UI 元素。"""

    text: str
    resource_id: str
    class_name: str
    content_desc: str
    bounds: list[int]          # [x1, y1, x2, y2]
    clickable: bool
    enabled: bool
    checkable: bool
    checked: bool
    focusable: bool

    @property
    def center(self) -> tuple[int, int]:
        x = (self.bounds[0] + self.bounds[2]) // 2
        y = (self.bounds[1] + self.bounds[3]) // 2
        return x, y

    @property
    def is_visible(self) -> bool:
        """零面积 bounds 的元素不会被渲染。"""
        w = self.bounds[2] - self.bounds[0]
        h = self.bounds[3] - self.bounds[1]
        return w > 0 and h > 0

    @property
    def label(self) -> str:
        """该元素最适合人类阅读的标签。"""
        return self.text or self.content_desc or self.resource_id.split("/")[-1]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "resource_id": self.resource_id,
            "class": self.class_name,
            "content_desc": self.content_desc,
            "bounds": self.bounds,
            "center": list(self.center),
            "clickable": self.clickable,
            "enabled": self.enabled,
        }


@dataclass
class ParsedScreen:
    """完整的页面解析结果。"""

    elements: list[UIElement] = field(default_factory=list)
    raw_xml: str = ""

    def clickable_elements(self) -> list[UIElement]:
        return [e for e in self.elements if e.clickable and e.is_visible]

    def visible_elements(self) -> list[UIElement]:
        """返回当前页面上可见的元素列表。"""
        return [e for e in self.elements if e.is_visible]

    def visible_labels(self, limit: int = 8) -> list[str]:
        """提取前若干个可见标签，便于错误诊断。"""
        labels: list[str] = []
        for el in self.visible_elements():
            label = el.label.strip()
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= limit:
                break
        return labels

    def find_by_text(self, text: str, exact: bool = False) -> UIElement | None:
        """按文字查找元素；优先精确匹配，避免短文本被误配到无关的更长文案。

        踩过的坑：星期切换栏的日期缩写（"一""二"等）都是单字，若直接走
        子串包含匹配，页面上任何包含该字符的无关文案（例如"换一批"刷新
        按钮里恰好带"一"字）都可能因为在元素列表中排在真正目标之前而被
        误命中，导致点击了完全不相关的按钮却没有任何报错。因此即使
        ``exact=False``（默认的"模糊匹配"模式），也应该先扫描一遍寻找
        完全相等的元素，找不到时才退化为子串包含匹配。
        """
        for el in self.elements:
            if el.text == text or el.content_desc == text:
                return el
        if exact:
            return None
        for el in self.elements:
            if text in el.text or text in el.content_desc:
                return el
        return None

    def find_by_resource_id(self, resource_id: str, index: int = 0) -> UIElement | None:
        """按 resource-id 查找第 index 个匹配元素（如 RecyclerView 中同 id 的多个卡片）。"""
        candidates = [
            el
            for el in self.elements
            if el.resource_id == resource_id or el.resource_id.endswith(resource_id)
        ]
        if 0 <= index < len(candidates):
            return candidates[index]
        return None

    def find_by_class_name(
        self,
        class_name: str,
        index: int = 0,
        clickable_only: bool = False,
    ) -> UIElement | None:
        """按 class name 查找第 N 个可见元素。"""
        candidates = [
            el
            for el in self.visible_elements()
            if el.class_name == class_name and (not clickable_only or el.clickable)
        ]
        if 0 <= index < len(candidates):
            return candidates[index]
        return None

    def to_agent_summary(self) -> dict:
        """供 Agent 消费的紧凑表示。"""
        visible = self.visible_elements()
        return {
            "element_count": len(visible),
            "clickable_count": len(self.clickable_elements()),
            "elements": [e.to_dict() for e in visible],
        }

    def bottom_overlay_elements(self) -> list[UIElement]:
        """启发式识别常驻在屏幕底部的悬浮控件本身（如底部 Tab 导航栏）。

        很多 App 的底部导航栏是独立于内容滚动区域的固定悬浮层：内容区域向上
        滚动后，紧贴屏幕底部的卡片/列表项可能仍有一部分被这类悬浮层遮挡，
        但 UIAutomator2 汇报的 bounds 只是布局坐标，并不反映视觉遮挡关系
        （元素在遮挡层下方依然会被判定为 ``is_visible``）。

        这里的启发式：紧贴当前已知最大 y2（近似屏幕高度）、横向跨度接近全屏
        宽度、且自身高度不算太大的容器类元素，视为候选悬浮层。返回悬浮层
        元素本身（而不只是边界坐标），供 ``is_occluded`` 把悬浮层自身从
        "是否被遮挡"的判断中排除——否则悬浮层会被判定为"被自己遮挡"，
        导致依赖遮挡检测收敛的微调滚动永远无法结束。
        """
        visible = self.visible_elements()
        if not visible:
            return []
        max_bottom = max(el.bounds[3] for el in visible)
        max_width = max(el.bounds[2] for el in visible)
        if max_bottom <= 0 or max_width <= 0:
            return []

        overlays: list[UIElement] = []
        for el in visible:
            x1, y1, x2, y2 = el.bounds
            width = x2 - x1
            height = y2 - y1
            touches_bottom = y2 >= max_bottom - 4
            wide_enough = width >= max_width * 0.6
            reasonable_height = 0 < height <= max_bottom * 0.25
            # 要求候选上方确实存在其他内容（另一元素完全位于候选顶部之上），
            # 避免页面元素很少、候选恰好是唯一/最靠下元素时被误判为悬浮层。
            has_content_above = any(
                other is not el and other.bounds[3] <= y1 for other in visible
            )
            if touches_bottom and wide_enough and reasonable_height and has_content_above:
                overlays.append(el)
        return overlays

    def bottom_overlay_top(self) -> int | None:
        """检测到的底部悬浮层中，最靠上的 top 坐标（"遮挡边界"）。"""
        overlays = self.bottom_overlay_elements()
        return min(el.bounds[1] for el in overlays) if overlays else None

    def is_occluded(self, element: UIElement) -> bool:
        """判断某元素是否与检测到的底部悬浮层存在重叠（可能被遮挡）。

        只要元素底边落在遮挡边界及以下（部分被遮挡或整体已滑到遮挡层下方），
        就认为存在遮挡风险——用 tap 解析出的坐标可能实际点在悬浮层上。
        悬浮层元素自身会被排除，不会被判定为"被自己遮挡"。
        """
        overlays = self.bottom_overlay_elements()
        if any(el is element for el in overlays):
            return False
        boundary = min((el.bounds[1] for el in overlays), default=None)
        if boundary is None:
            return False
        return element.bounds[3] > boundary

    def has_meaningful_content(self, min_text_elements: int = 1) -> bool:
        """当 XML 树信息足够丰富、可跳过 VLM 兜底时返回 True。

        满足以下任一条件即认为 XML 足够：
          - 至少存在一个可点击元素（即使没有文本，action_tool 也能定位
            按钮 / 输入框），或
          - 携带 text 或 content_desc 的可见元素数量达到
            ``min_text_elements``。

        设置 ``min_text_elements=0`` 表示始终信任 XML（不自动触发 VLM）。
        """
        visible = self.visible_elements()
        has_clickable = any(e.clickable for e in visible)
        if has_clickable:
            return True
        with_text = [e for e in visible if e.text or e.content_desc]
        return len(with_text) >= min_text_elements


def parse_page_source(xml_str: str) -> ParsedScreen:
    """将 Appium page_source XML 解析为 ParsedScreen。

    Args:
        xml_str: 来自 driver.page_source 的原始 XML 字符串。

    Returns:
        包含所有已提取 UIElement 的 ParsedScreen。
    """
    if not xml_str or not xml_str.strip():
        return ParsedScreen(raw_xml=xml_str)

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ParsedScreen(raw_xml=xml_str)

    elements: list[UIElement] = []
    _walk(root, elements)

    return ParsedScreen(elements=elements, raw_xml=xml_str)


def compute_structure_hash(xml_str: str) -> str:
    """根据 XML 结构计算稳定哈希（忽略动态文本内容）。

    使用元素标签名和 resource-id，而不是文本内容，因此文本更新
    （例如计数器变化）后的页面仍可命中同一个缓存条目。
    """
    import hashlib

    if not xml_str:
        return ""
    try:
        root = ET.fromstring(xml_str)
        parts: list[str] = []
        for el in root.iter():
            tag = el.tag
            res_id = el.get("resource-id", "")
            parts.append(f"{tag}|{res_id}")
        fingerprint = ";".join(parts)
        return hashlib.md5(fingerprint.encode()).hexdigest()
    except ET.ParseError:
        return ""


# ── 内部辅助方法 ─────────────────────────────────────────────────────────────

def _walk(node: ET.Element, out: list[UIElement]) -> None:
    """从 XML 树中递归提取 UIElement。"""
    el = _extract_element(node)
    if el is not None:
        out.append(el)
    for child in node:
        _walk(child, out)


def _extract_element(node: ET.Element) -> UIElement | None:
    """从单个 XML 节点提取 UIElement；无可用信息时返回 None。"""
    text = (node.get("text") or "").strip()
    content_desc = (node.get("content-desc") or "").strip()
    resource_id = node.get("resource-id", "")
    class_name = node.tag  # 在 UIAutomator2 中，标签名就是 class name

    # 跳过没有识别信息的节点
    if not text and not content_desc and not resource_id:
        return None

    bounds_str = node.get("bounds", "")
    bounds = _parse_bounds(bounds_str)

    return UIElement(
        text=text,
        resource_id=resource_id,
        class_name=class_name,
        content_desc=content_desc,
        bounds=bounds,
        clickable=node.get("clickable", "false").lower() == "true",
        enabled=node.get("enabled", "true").lower() == "true",
        checkable=node.get("checkable", "false").lower() == "true",
        checked=node.get("checked", "false").lower() == "true",
        focusable=node.get("focusable", "false").lower() == "true",
    )


def _parse_bounds(bounds_str: str) -> list[int]:
    """解析 "[x1,y1][x2,y2]" 为 [x1, y1, x2, y2]。"""
    nums = re.findall(r"\d+", bounds_str)
    return [int(n) for n in nums] if len(nums) == 4 else [0, 0, 0, 0]
