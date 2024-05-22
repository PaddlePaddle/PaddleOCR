from typing import Any

import pytest
from paddleocr import PaddleOCR, PPStructure


# Test image paths
IMAGE_PATHS_OCR = ["./doc/imgs_en/254.jpg", "./doc/imgs_en/img_10.jpg"]
IMAGE_PATHS_STRUCTURE = [
    "./ppstructure/docs/table/layout.jpg",
    "./ppstructure/docs/table/1.png",
]


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
