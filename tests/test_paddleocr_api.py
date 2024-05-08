import pytest
import cv2
from paddleocr import PaddleOCR


@pytest.fixture
def ocr():
    return PaddleOCR(lang="ch")


@pytest.fixture
def img():
    return cv2.imread("doc/imgs/1.jpg")


@pytest.mark.parametrize(
    "det, rec", [(True, True), (True, False), (False, True), (False, False)]
)
def test_ocr_det_rec_api(ocr, img, det, rec):
    result = ocr.ocr(img, det=det, rec=rec)
    assert result is not None
    assert isinstance(result, list)


def test_ocr_fp16():
    ocr = PaddleOCR(lang="ch", use_angle_cls=True, precision=True)
    img = cv2.imread("doc/imgs/1.jpg")
    result = ocr.ocr(img, det=True, rec=True)
    assert result is not None
    assert isinstance(result, list)


def test_ocr_with_none_image():
    ocr = PaddleOCR(lang="ch", use_angle_cls=True)
    img = None
    with pytest.raises(AssertionError):
        ocr.ocr(img, det=True, rec=True)
