# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Union, Optional

from .base import ConvertResult
from .registry import default_registry

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
        if isinstance(e, (FileNotFoundError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to convert {file_path.name}: {e}") from e

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


def supported_formats() -> list[str]:
    """Return a list of supported file extensions."""
    return default_registry.supported_extensions()
