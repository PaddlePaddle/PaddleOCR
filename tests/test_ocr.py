# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import Any

import pytest

from paddleocr import PaddleOCR

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

IMAGE_PATHS_OCR = [str(test_file_dir / "254.jpg"), str(test_file_dir / "img_10.jpg")]


def _check_ocr_result(ocr_result: list, image_path: str):
    assert ocr_result is not None
    assert isinstance(ocr_result, list)
    assert len(ocr_result) == 1
    assert isinstance(ocr_result[0], dict)
    res = ocr_result[0]
    assert isinstance(res, dict)
    assert len(res["dt_polys"]) > 0
    assert isinstance(res["rec_texts"], list)
    assert len(res["rec_texts"]) > 0
    for text in res["rec_texts"]:
        assert isinstance(text, str)
    assert Path(res["input_path"]).samefile(image_path)


@pytest.fixture(scope="module")
def ocr_engine() -> PaddleOCR:
    return PaddleOCR()


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
def test_predict_basic(ocr_engine: PaddleOCR, image_path: str) -> None:
    """
    Test PaddleOCR's basic OCR functionality.

    Args:
        ocr_engine: An instance of `PaddleOCR`.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.predict(image_path)

    _check_ocr_result(result, image_path)

    res = result[0]
    assert (
        res["doc_preprocessor_res"]["model_settings"]["use_doc_orientation_classify"]
        is True
    )
    assert res["doc_preprocessor_res"]["model_settings"]["use_doc_unwarping"] is True
    assert res["model_settings"]["use_textline_orientation"] is True


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
@pytest.mark.parametrize("use_doc_orientation_classify", [True, False, None])
@pytest.mark.parametrize("use_doc_unwarping", [True, False, None])
@pytest.mark.parametrize("use_textline_orientation", [True, False, None])
def test_predict_model_settings(
    ocr_engine: PaddleOCR,
    image_path: str,
    use_doc_orientation_classify: bool | None,
    use_doc_unwarping: bool | None,
    use_textline_orientation: bool | None,
) -> None:
    """Test PaddleOCR's OCR functionality with different model settings."""
    result = ocr_engine.predict(
        image_path,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )

    _check_ocr_result(result, image_path)

    res = result[0]
    if use_doc_orientation_classify is None:
        use_doc_orientation_classify = True
    if use_doc_unwarping is None:
        use_doc_unwarping = True
    if use_textline_orientation is None:
        use_textline_orientation = True
    if res["model_settings"]["use_doc_preprocessor"]:
        assert (
            res["doc_preprocessor_res"]["model_settings"][
                "use_doc_orientation_classify"
            ]
            is use_doc_orientation_classify
        )
        assert (
            res["doc_preprocessor_res"]["model_settings"]["use_doc_unwarping"]
            is use_doc_unwarping
        )
    assert res["model_settings"]["use_textline_orientation"] is use_textline_orientation


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
@pytest.mark.parametrize(
    "params",
    [
        {"text_det_limit_side_len": 640, "text_det_limit_type": "min"},
        {"text_det_thresh": 0.5},
        {"text_det_box_thresh": 0.3},
        {"text_det_unclip_ratio": 3.0},
        {"text_rec_score_thresh": 0.5},
    ],
)
def test_predict_det_rec_params(
    ocr_engine: PaddleOCR, image_path: str, params: dict
) -> None:
    """
    Test PaddleOCR's OCR functionality with different text detection and text
    recognition parameters.
    """
    result = ocr_engine.predict(image_path, **params)

    _check_ocr_result(result, image_path)

    res = result[0]
    if "text_det_limit_side_len" in params:
        assert (
            res["text_det_params"]["limit_side_len"]
            == params["text_det_limit_side_len"]
        )
    if "text_det_limit_type" in params:
        assert res["text_det_params"]["limit_type"] == params["text_det_limit_type"]
    if "text_det_thresh" in params:
        assert res["text_det_params"]["thresh"] == params["text_det_thresh"]
    if "text_det_box_thresh" in params:
        assert res["text_det_params"]["box_thresh"] == params["text_det_box_thresh"]
    if "text_det_unclip_ratio" in params:
        assert res["text_det_params"]["unclip_ratio"] == params["text_det_unclip_ratio"]
    if "text_rec_score_thresh" in params:
        assert res["text_rec_score_thresh"] == params["text_rec_score_thresh"]


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
