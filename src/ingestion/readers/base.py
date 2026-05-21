"""Abstract base class for document format readers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """Parsed document with extracted text and metadata."""

    id: str
    source_path: str
    content: str
    metadata: dict = field(default_factory=dict)
    # metadata includes: title, author, page_count, sections, tables, etc.


class BaseReader(ABC):
    """Abstract reader for a specific file format."""

    @abstractmethod
    def read(self, path: Path) -> Document:
        """Parse a file into a Document. Raises IngestionError on failure."""
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this reader handles."""
        ...
