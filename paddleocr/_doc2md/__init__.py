"""paddleocr._doc2md - Convert office documents to Markdown."""

from .core import convert, supported_formats
from .base import ConvertResult, BaseConverter
from .registry import default_registry

__all__ = [
    "convert",
    "supported_formats",
    "ConvertResult",
    "BaseConverter",
    "default_registry",
]
