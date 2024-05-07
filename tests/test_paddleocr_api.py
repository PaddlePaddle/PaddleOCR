import pytest
import cv2
from paddleocr import PaddleOCR


@pytest.fixture
def ocr():
    return PaddleOCR(lang="en")


@pytest.fixture
def img():
    return cv2.imread("doc/imgs/1.jpg")


def test_ocr_with_detection_and_recognition(ocr, img):
    result = ocr.ocr(img, det=True, rec=True)
    assert result is not None
    assert isinstance(result, list)


def test_ocr_with_detection_only(ocr, img):
    result = ocr.ocr(img, det=True, rec=False)
    assert result is not None
    assert isinstance(result, list)


def test_ocr_with_recognition_only(ocr, img):
    result = ocr.ocr(img, det=False, rec=True)
    assert result is not None
    assert isinstance(result, list)


def test_ocr_without_detection_and_recognition(ocr, img):
    result = ocr.ocr(img, det=False, rec=False)
    assert result is not None
    assert isinstance(result, list)
