import mimetypes
from pathlib import Path
from typing import Type

from .base import BaseConverter


class ConverterRegistry:
    """Registry mapping file extensions and MIME types to converter classes."""

    def __init__(self):
        self._ext_map: dict[str, Type[BaseConverter]] = {}
        self._mime_map: dict[str, Type[BaseConverter]] = {}

    def register(self, converter_cls: Type[BaseConverter]) -> Type[BaseConverter]:
        """Register a converter class; can be used as a decorator."""
        for ext in converter_cls.supported_extensions:
            self._ext_map[ext.lower().lstrip(".")] = converter_cls
        for mime in converter_cls.supported_mimetypes:
            self._mime_map[mime] = converter_cls
        return converter_cls

    def get_converter(self, file_path: Path) -> BaseConverter:
        """Return an appropriate converter instance for the given file path."""
        ext = file_path.suffix.lower().lstrip(".")
        if ext in self._ext_map:
            return self._ext_map[ext]()

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type in self._mime_map:
            return self._mime_map[mime_type]()

        supported = ", ".join(f".{e}" for e in sorted(self._ext_map.keys()))
        raise ValueError(f"Unsupported format: .{ext}\nSupported formats: {supported}")

    def supported_extensions(self) -> list[str]:
        return sorted(self._ext_map.keys())


# Global singleton registry
default_registry = ConverterRegistry()
