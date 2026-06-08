"""支持章节抽取的 Markdown 文档 reader。"""

import re
import uuid
from pathlib import Path

from src.core.exceptions import IngestionError
from src.ingestion.readers.base import BaseReader, Document


class MarkdownReader(BaseReader):
    def read(self, path: Path) -> Document:
        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding="gbk", errors="replace")

        sections = self._extract_sections(raw)
        metadata: dict = {
            "sections": [s["title"] for s in sections if s["title"]],
        }

        # 尝试抽取 YAML frontmatter
        content = raw
        fm_match = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
        if fm_match:
            metadata["frontmatter"] = fm_match.group(1)
            content = raw[fm_match.end() :]

        return Document(
            id=str(uuid.uuid4()),
            source_path=str(path.resolve()),
            content=content,
            metadata=metadata,
        )

    def _extract_sections(self, text: str) -> list[dict]:
        """从 markdown 中抽取标题层级。"""
        sections: list[dict] = []
        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                sections.append({
                    "level": len(m.group(1)),
                    "title": m.group(2).strip(),
                })
        return sections

    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]
