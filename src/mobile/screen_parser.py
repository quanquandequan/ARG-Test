"""Parse Appium page_source XML into a structured element list.

Appium Android (UIAutomator2) returns XML like:
  <hierarchy>
    <android.widget.FrameLayout bounds="[0,0][1080,2340]">
      <android.widget.Button text="登录" resource-id="com.pkg:id/login_btn"
        bounds="[200,500][880,570]" clickable="true"/>
    </android.widget.FrameLayout>
  </hierarchy>

We extract only the elements useful to a test agent:
  - Has text, content-desc, or a known resource-id
  - Visible (bounds not collapsed to a zero-area rect)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class UIElement:
    """A single UI element extracted from the page source."""

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
        """Element with zero-area bounds is not rendered."""
        w = self.bounds[2] - self.bounds[0]
        h = self.bounds[3] - self.bounds[1]
        return w > 0 and h > 0

    @property
    def label(self) -> str:
        """Best human-readable label for this element."""
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
    """The full parsed page."""

    elements: list[UIElement] = field(default_factory=list)
    raw_xml: str = ""

    def clickable_elements(self) -> list[UIElement]:
        return [e for e in self.elements if e.clickable and e.is_visible]

    def find_by_text(self, text: str, exact: bool = False) -> UIElement | None:
        for el in self.elements:
            if exact:
                if el.text == text or el.content_desc == text:
                    return el
            else:
                if text in el.text or text in el.content_desc:
                    return el
        return None

    def find_by_resource_id(self, resource_id: str) -> UIElement | None:
        for el in self.elements:
            if el.resource_id == resource_id or el.resource_id.endswith(resource_id):
                return el
        return None

    def to_agent_summary(self) -> dict:
        """Compact representation for Agent consumption."""
        visible = [e for e in self.elements if e.is_visible]
        return {
            "element_count": len(visible),
            "clickable_count": len(self.clickable_elements()),
            "elements": [e.to_dict() for e in visible],
        }

    def has_meaningful_content(self, min_elements: int = 3) -> bool:
        """True if XML tree has enough info to avoid needing VLM fallback."""
        visible = [e for e in self.elements if e.is_visible]
        with_text = [e for e in visible if e.text or e.content_desc]
        return len(with_text) >= min_elements


def parse_page_source(xml_str: str) -> ParsedScreen:
    """Parse Appium page_source XML into a ParsedScreen.

    Args:
        xml_str: Raw XML string from driver.page_source.

    Returns:
        ParsedScreen containing all extracted UIElements.
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
    """Compute a stable hash from XML structure (ignores dynamic text content).

    Uses element tag names and resource IDs, not text content, so that a
    page with updated text (e.g. a counter) still hits the same cache entry.
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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _walk(node: ET.Element, out: list[UIElement]) -> None:
    """Recursively extract UIElements from the XML tree."""
    el = _extract_element(node)
    if el is not None:
        out.append(el)
    for child in node:
        _walk(child, out)


def _extract_element(node: ET.Element) -> UIElement | None:
    """Extract a UIElement from a single XML node; return None if not useful."""
    text = (node.get("text") or "").strip()
    content_desc = (node.get("content-desc") or "").strip()
    resource_id = node.get("resource-id", "")
    class_name = node.tag  # in UIAutomator2 the tag IS the class name

    # Skip nodes with no identifying information
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
    """Parse "[x1,y1][x2,y2]" → [x1, y1, x2, y2]."""
    nums = re.findall(r"\d+", bounds_str)
    return [int(n) for n in nums] if len(nums) == 4 else [0, 0, 0, 0]
