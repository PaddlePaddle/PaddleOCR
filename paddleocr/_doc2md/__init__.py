"""paddleocr._doc2md - Convert office documents to Markdown."""

from .core import convert, convert_bytes, supported_formats
from .base import ConvertResult, BaseConverter
from .registry import default_registry
from .exceptions import Any2MDError, UnsupportedFormatError, ConversionError

__all__ = [
    "convert",
    "convert_bytes",
    "supported_formats",
    "ConvertResult",
    "BaseConverter",
    "default_registry",
    "Any2MDError",
    "UnsupportedFormatError",
    "ConversionError",
]
