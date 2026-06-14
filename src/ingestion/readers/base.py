"""文档格式 reader 抽象基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """解析后的文档，包含抽取文本和元数据。"""

    id: str
    source_path: str
    content: str
    metadata: dict = field(default_factory=dict)
    # metadata 包含 title、author、page_count、sections、tables 等。
    # segments 非空时，每个元素为 {content: str, metadata: dict}，
    # 代表天然边界文本块（如 Excel 行），IngestionPipeline 会直接一段一 chunk。
    segments: list[dict] = field(default_factory=list)


class BaseReader(ABC):
    """特定文件格式 reader 的抽象类。"""

    @abstractmethod
    def read(self, path: Path) -> Document:
        """将文件解析为 Document；失败时抛出 IngestionError。"""
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """返回该 reader 支持的文件扩展名列表。"""
        ...
