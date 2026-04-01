from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import tempfile


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

    def convert_bytes(
        self, data: bytes, original_filename: str = "", **kwargs
    ) -> ConvertResult:
        """Convert from bytes (e.g. web upload scenarios)."""
        suffix = Path(original_filename).suffix if original_filename else ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return self.convert_file(tmp_path, **kwargs)
        finally:
            tmp_path.unlink(missing_ok=True)
