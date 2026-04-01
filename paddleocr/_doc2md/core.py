from pathlib import Path
from typing import Union, Optional

from .base import ConvertResult
from .registry import default_registry
from .exceptions import ConversionError

# Trigger registration of all built-in converters
from . import converters  # noqa: F401


def convert(
    source: Union[str, Path],
    *,
    output: Optional[Union[str, Path]] = None,
    **kwargs,
) -> ConvertResult:
    """
    Convert an office document to Markdown.

    Args:
        source: Path to the source file.
        output: Optional output file path. If provided, Markdown is written there.
        **kwargs: Extra arguments forwarded to the specific converter.

    Returns:
        ConvertResult object.

    Examples:
        >>> from paddleocr import doc2md_convert
        >>> result = doc2md_convert("report.docx")
        >>> print(result.markdown)
    """
    file_path = Path(source)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    converter = default_registry.get_converter(file_path)

    try:
        result = converter.convert_file(file_path, **kwargs)
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ConversionError)):
            raise
        raise ConversionError(f"Failed to convert {file_path.name}: {e}") from e

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.markdown, encoding="utf-8")
        if result.images:
            images_dir = output_path.parent / "images"
            images_dir.mkdir(exist_ok=True)
            for rel_path, img_bytes in result.images.items():
                img_file = output_path.parent / rel_path
                img_file.write_bytes(img_bytes)

    return result


def convert_bytes(data: bytes, filename: str, **kwargs) -> ConvertResult:
    """
    Convert from raw bytes (e.g. web upload scenarios).

    Args:
        data: Raw file bytes.
        filename: Original filename used to determine the format.
    """
    fake_path = Path(filename)
    converter = default_registry.get_converter(fake_path)
    return converter.convert_bytes(data, original_filename=filename, **kwargs)


def supported_formats() -> list[str]:
    """Return a list of supported file extensions."""
    return default_registry.supported_extensions()
