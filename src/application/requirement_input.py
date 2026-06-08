"""需求输入解析：统一处理文本和本地需求文件。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.core.exceptions import IngestionError
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".xlsx", ".xlsm", ".xmind"}
_PATH_PATTERN = re.compile(
    r"(?P<path>(?:/|\.{1,2}/|[\w.-]+/)[^\s，,。；;：:\"'）)]+"
    r"(?:\.txt|\.md|\.pdf|\.xlsx|\.xlsm|\.xmind))",
    re.IGNORECASE,
)
_COMMAND_MARKERS = ("读取", "分析", "文件", "需求", "内容", "请", "帮我")


@dataclass(slots=True, frozen=True)
class RequirementInput:
    """解析后的需求输入。"""

    content: str
    source_path: str

    @property
    def is_file(self) -> bool:
        return bool(self.source_path)


class RequirementInputError(ValueError):
    """需求输入不可用。"""


def resolve_requirement_input(
    *,
    requirement: str = "",
    requirement_file: str = "",
    loader: DocumentLoader | None = None,
    cleaner: TextCleaner | None = None,
    cwd: Path | None = None,
) -> RequirementInput:
    """将需求文本或文件路径解析为真实需求正文。"""
    raw_requirement = (requirement or "").strip()
    file_path = _find_requirement_file(
        requirement=raw_requirement,
        requirement_file=(requirement_file or "").strip(),
        cwd=cwd or Path.cwd(),
    )
    if file_path is not None:
        return _load_requirement_file(
            file_path,
            loader=loader or DocumentLoader(),
            cleaner=cleaner or TextCleaner(),
        )

    content = raw_requirement.strip()
    if not content:
        raise RequirementInputError("请提供需求文档内容或可读取的需求文件路径。")
    return RequirementInput(content=content, source_path="")


def _find_requirement_file(
    *,
    requirement: str,
    requirement_file: str,
    cwd: Path,
) -> Path | None:
    if requirement_file:
        return _normalise_candidate_path(requirement_file, cwd)

    exact_path = _normalise_candidate_path(requirement, cwd)
    if exact_path is not None:
        return exact_path

    matches = list(_PATH_PATTERN.finditer(requirement))
    if len(matches) != 1 or not _looks_like_file_command(requirement, matches[0]):
        return None
    return _normalise_candidate_path(matches[0].group("path"), cwd)


def _looks_like_file_command(text: str, match: re.Match[str]) -> bool:
    """仅对短的“读取/分析文件”指令抽取路径，避免误读长 PRD 正文里的路径。"""
    residue = (text[:match.start()] + text[match.end():]).strip()
    if len(text) > 240:
        return False
    compact = re.sub(r"\s+", "", residue)
    return not compact or any(marker in compact for marker in _COMMAND_MARKERS)


def _normalise_candidate_path(value: str, cwd: Path) -> Path | None:
    candidate = value.strip().strip("\"'“”‘’")
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = cwd / path
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        return None
    return path


def _load_requirement_file(
    path: Path,
    *,
    loader: DocumentLoader,
    cleaner: TextCleaner,
) -> RequirementInput:
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise RequirementInputError(
            f"不支持的需求文件类型：{path.suffix}。"
            f"支持格式：{', '.join(sorted(_SUPPORTED_SUFFIXES))}"
        )
    try:
        doc = loader.load(path)
    except IngestionError as exc:
        raise RequirementInputError(f"无法读取需求文件：{exc}") from exc

    content = cleaner.clean(doc.content).strip()
    if not content:
        raise RequirementInputError(f"需求文件内容为空：{path}")
    return RequirementInput(content=content, source_path=str(path.resolve()))
