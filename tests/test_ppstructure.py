# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import Any

import pytest

from paddleocr import PPStructure

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

IMAGE_PATHS_STRUCTURE = [
    str(test_file_dir / "ppstructure" / "layout.jpg"),
    str(test_file_dir / "ppstructure" / "1.png"),
]


@pytest.fixture(params=["en", "ch"])
def structure_engine(request: Any) -> PPStructure:
    """
    Initialize PPStructure engine with different languages.

    Args:
        request: pytest fixture request object.

    Returns:
        An instance of PPStructure.
    """
    return PPStructure(lang=request.param)


def test_structure_initialization(structure_engine: PPStructure) -> None:
    """
    Test PPStructure initialization.

    Args:
        structure_engine: An instance of PPStructure.
    """
    assert structure_engine is not None


@pytest.mark.parametrize("image_path", IMAGE_PATHS_STRUCTURE)
def test_structure_function(structure_engine: PPStructure, image_path: str) -> None:
    """
    Test PPStructure structure analysis functionality with different images.

    Args:
        structure_engine: An instance of PPStructure.
        image_path: Path to the image to be processed.
    """
    result = structure_engine(image_path)
    assert result is not None
    assert isinstance(result, list)
