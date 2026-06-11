# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import pytest

from paddleocr import PaddleOCR
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def ocr_engine() -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
    )


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
    ocr_engine = PaddleOCR(lang="ch", ocr_version="PP-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "PP-OCRv5_server_rec"
    ocr_engine = PaddleOCR(lang="chinese_cht", ocr_version="PP-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "PP-OCRv5_server_rec"
    ocr_engine = PaddleOCR(lang="en", ocr_version="PP-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"
    ocr_engine = PaddleOCR(lang="japan", ocr_version="PP-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "PP-OCRv5_server_rec"
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


def test_pp_ocrv6_model_names():
    engine = object.__new__(PaddleOCR)
    for lang in ("ch", "chinese_cht", "en", "japan", "fr", "de", "vi", "ku", "az"):
        det, rec = engine._get_ocr_model_names(lang, "PP-OCRv6")
        assert det == "PP-OCRv6_medium_det"
        assert rec == "PP-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("pi", "PP-OCRv6")
    assert det is None
    assert rec is None
    det, rec = engine._get_ocr_model_names("ru", "PP-OCRv6")
    assert det is None
    assert rec is None


def test_default_ocr_model_names():
    engine = object.__new__(PaddleOCR)
    det, rec = engine._get_ocr_model_names(None, None)
    assert det == "PP-OCRv6_medium_det"
    assert rec == "PP-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("ch", None)
    assert det == "PP-OCRv6_medium_det"
    assert rec == "PP-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("fr", None)
    assert det == "PP-OCRv6_medium_det"
    assert rec == "PP-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("ru", None)
    assert det == "PP-OCRv5_server_det"
    assert rec == "eslav_PP-OCRv5_mobile_rec"
    det, rec = engine._get_ocr_model_names("az", None)
    assert det == "PP-OCRv6_medium_det"
    assert rec == "PP-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("ku", None)
    assert det == "PP-OCRv6_medium_det"
    assert rec == "PP-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("pi", None)
    assert det == "PP-OCRv5_server_det"
    assert rec == "latin_PP-OCRv5_mobile_rec"


def test_unsupported_lang_version_raises():
    with pytest.raises(ValueError, match="No models are available"):
        PaddleOCR(lang="pi", ocr_version="PP-OCRv6")
    with pytest.raises(ValueError, match="No models are available"):
        PaddleOCR(lang="ru", ocr_version="PP-OCRv6")
