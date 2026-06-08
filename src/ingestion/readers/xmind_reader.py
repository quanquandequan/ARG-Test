"""XMind 思维导图 reader：从 .xmind 文件抽取主题树。

XMind 文件是 ZIP 归档，包含 content.json（XMind Zen / 2020+）和/或
content.xml（旧版 XMind 8）。两者同时存在时优先使用 JSON，因为 XML 副本
通常是 XMind 双格式导出留下的陈旧模板产物。
"""

import json as _json
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from src.core.exceptions import IngestionError
from src.ingestion.readers.base import BaseReader, Document

_NS = {"fo": "http://www.w3.org/1999/XSL/Format", "svg": "http://www.w3.org/2000/svg"}
_XML_CORRUPT_SENTINEL = b"This file can not be opened normally"


class XmindReader(BaseReader):
    def read(self, path: Path) -> Document:
        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                has_json = "content.json" in names
                has_xml = "content.xml" in names

                if not has_json and not has_xml:
                    raise IngestionError(
                        f"Unrecognized XMind format: no content.xml or content.json in {path}"
                    )

                # 优先使用 JSON（XMind Zen / 2020+）；两种格式同时存在时，
                # JSON 始终是真实数据源，XML 副本通常是陈旧模板。
                if has_json:
                    json_bytes = zf.read("content.json")
                    if len(json_bytes) > 500:
                        return self._read_zen(json_bytes, path)

                # 兜底使用 XML；如果是损坏警告模板则跳过
                if has_xml:
                    xml_bytes = zf.read("content.xml")
                    if _XML_CORRUPT_SENTINEL in xml_bytes:
                        raise IngestionError(
                            f"XMind file has corrupted XML and no valid JSON content: {path}"
                        )
                    root = ET.fromstring(xml_bytes)
                    title, lines = self._extract_sheet(root)
                    content = "\n".join(lines) if lines else ""
                    return Document(
                        id=str(uuid.uuid4()),
                        source_path=str(path.resolve()),
                        content=content,
                        metadata={"title": title, "format": "xmind"},
                    )

                raise IngestionError(f"No valid content found in XMind: {path}")

        except zipfile.BadZipFile as e:
            raise IngestionError(f"Invalid XMind file (not a ZIP): {path}") from e
        except IngestionError:
            raise
        except ET.ParseError as e:
            raise IngestionError(f"Failed to parse XMind XML: {path} — {e}") from e
        except Exception as e:
            raise IngestionError(f"Failed to read XMind: {path} — {e}") from e

    def _read_zen(self, json_bytes: bytes, path: Path) -> Document:
        data = _json.loads(json_bytes)
        title = ""
        lines: list[str] = []

        for sheet in data if isinstance(data, list) else [data]:
            root_topic = sheet.get("rootTopic", {})
            if not root_topic:
                continue
            title = root_topic.get("title", sheet.get("title", ""))
            self._walk_topic_zen(root_topic, lines, depth=0)

        content = "\n".join(lines) if lines else ""
        return Document(
            id=str(uuid.uuid4()),
            source_path=str(path.resolve()),
            content=content,
            metadata={"title": title, "format": "xmind"},
        )

    def _walk_topic_zen(self, topic: dict, lines: list[str], depth: int) -> str:
        title = topic.get("title", "")
        indent = "  " * depth
        lines.append(f"{indent}- {title}")

        notes = topic.get("notes", {})
        if isinstance(notes, dict):
            plain = notes.get("plain", {})
            note_text = (
                plain.get("content", "") if isinstance(plain, dict) else str(plain)
            ).strip()
            if note_text:
                lines.append(f"{indent}  > {note_text}")

        for child in topic.get("children", {}).get("attached", []):
            self._walk_topic_zen(child, lines, depth + 1)

        return title

    def _extract_sheet(self, sheet_elem: ET.Element) -> tuple[str, list[str]]:
        title = ""
        lines: list[str] = []

        for topic in sheet_elem.iter("topic"):
            topic_title = topic.findtext("title", default="").strip()
            if not topic_title:
                continue
            if not title:
                title = topic_title

            prefix = self._ancestor_path(topic)
            indent = "  " * max(len(prefix) - 1, 0)
            lines.append(f"{indent}- {topic_title}")

            notes_elem = topic.find("notes")
            if notes_elem is not None:
                for plain in notes_elem.iter("plain"):
                    note_text = (plain.text or "").strip()
                    if note_text:
                        lines.append(f"{indent}  > {note_text}")

            labels_elem = topic.find("labels")
            if labels_elem is not None:
                label_texts = [
                    (lbl.text or "").strip()
                    for lbl in labels_elem.findall("label")
                    if lbl.text
                ]
                if label_texts:
                    lines.append(f"{indent}  [labels: {', '.join(label_texts)}]")

        return title, lines

    def _ancestor_path(self, elem: ET.Element) -> list[str]:
        path: list[str] = []
        current = elem
        while current is not None:
            title_elem = current.find("title")
            if title_elem is not None and title_elem.text:
                path.append(title_elem.text.strip())
            current = self._parent_topic(current)
        return list(reversed(path))

    def _parent_topic(self, elem: ET.Element) -> ET.Element | None:
        parent = elem
        while parent is not None:
            parent = parent.find("..")
            if parent is not None and parent.tag == "topic":
                return parent
        return None

    def supported_extensions(self) -> list[str]:
        return [".xmind"]
