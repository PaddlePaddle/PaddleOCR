from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConvertResult:
    """Conversion result."""

    markdown: str
    title: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)  # {relative_path: image_bytes}


class BaseConverter(ABC):
    """Abstract base class for all format converters."""

    supported_extensions: list[str] = []
    supported_mimetypes: list[str] = []

    @abstractmethod
    def convert_file(self, file_path: Path, **kwargs) -> ConvertResult:
        """Convert a file to Markdown."""
        ...
