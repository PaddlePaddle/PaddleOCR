import pytest

from paddleocr import PaddleOCR
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def ocr_engine() -> PaddleOCR:
    return PaddleOCR()


# TODO: Should we separate unit tests and integration tests?
@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_predict(ocr_engine: PaddleOCR, image_path: str) -> None:
    """
    Test PaddleOCR's OCR functionality.

    Args:
        ocr_engine: An instance of `PaddleOCR`.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    assert len(res["dt_polys"]) > 0
    assert isinstance(res["rec_texts"], list)
    assert len(res["rec_texts"]) > 0
    for text in res["rec_texts"]:
        assert isinstance(text, str)


# TODO: Also check passing `None`
@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_textline_orientation": False},
        {"text_det_limit_side_len": 640, "text_det_limit_type": "min"},
        {"text_det_thresh": 0.5},
        {"text_det_box_thresh": 0.3},
        {"text_det_unclip_ratio": 3.0},
        {"text_rec_score_thresh": 0.5},
    ],
)
def test_predict_params(
    monkeypatch,
    ocr_engine: PaddleOCR,
    params: dict,
) -> None:
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        ocr_engine,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test init params


def test_lang_and_ocr_version():
    ocr_engine = PaddleOCR(lang="ch", ocr_version="PP-OCRv4")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv4_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "PP-OCRv4_mobile_rec"
    ocr_engine = PaddleOCR(lang="en", ocr_version="PP-OCRv4")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv4_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "en_PP-OCRv4_mobile_rec"
    ocr_engine = PaddleOCR(lang="ch", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "PP-OCRv3_mobile_rec"
    ocr_engine = PaddleOCR(lang="en", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "en_PP-OCRv3_mobile_rec"
    ocr_engine = PaddleOCR(lang="fr", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"] == "latin_PP-OCRv3_mobile_rec"
    )
    ocr_engine = PaddleOCR(lang="ar", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"]
        == "arabic_PP-OCRv3_mobile_rec"
    )
    ocr_engine = PaddleOCR(lang="ru", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"]
        == "cyrillic_PP-OCRv3_mobile_rec"
    )
    ocr_engine = PaddleOCR(lang="hi", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"]
        == "devanagari_PP-OCRv3_mobile_rec"
    )
    ocr_engine = PaddleOCR(lang="japan", ocr_version="PP-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"] == "japan_PP-OCRv3_mobile_rec"
    )
