# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import Any

import pytest

from paddleocr import PaddleOCR

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

IMAGE_PATHS_OCR = [str(test_file_dir / "254.jpg"), str(test_file_dir / "img_10.jpg")]


@pytest.fixture(params=["en", "ch"])
def ocr_engine(request: Any) -> PaddleOCR:
    """
    Initialize PaddleOCR engine with different languages.

    Args:
        request: pytest fixture request object.

    Returns:
        An instance of PaddleOCR.
    """
    return PaddleOCR(lang=request.param)


def test_ocr_initialization(ocr_engine: PaddleOCR) -> None:
    """
    Test PaddleOCR initialization.

    Args:
        ocr_engine: An instance of PaddleOCR.
    """
    assert ocr_engine is not None


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
def test_ocr_function(ocr_engine: PaddleOCR, image_path: str) -> None:
    """
    Test PaddleOCR OCR functionality with different images.

    Args:
        ocr_engine: An instance of PaddleOCR.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.ocr(image_path)
    assert result is not None
    assert isinstance(result, list)


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
def test_ocr_det_only(ocr_engine: PaddleOCR, image_path: str) -> None:
    """
    Test PaddleOCR OCR functionality with detection only.

    Args:
        ocr_engine: An instance of PaddleOCR.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.ocr(image_path, det=True, rec=False)
    assert result is not None
    assert isinstance(result, list)


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
def test_ocr_rec_only(ocr_engine: PaddleOCR, image_path: str) -> None:
    """
    Test PaddleOCR OCR functionality with recognition only.

    Args:
        ocr_engine: An instance of PaddleOCR.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.ocr(image_path, det=False, rec=True)
    assert result is not None
    assert isinstance(result, list)
