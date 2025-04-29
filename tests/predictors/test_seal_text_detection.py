import pytest

from paddleocr import SealTextDetection
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def seal_text_detection_predictor():
    return SealTextDetection()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "seal.png",
    ],
)
def test_predict(seal_text_detection_predictor, image_path):
    result = seal_text_detection_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "dt_polys",
        "dt_scores",
    }


@pytest.mark.parametrize(
    "params",
    [
        {"limit_side_len": 640, "limit_type": "min"},
        {"thresh": 0.5},
        {"box_thresh": 0.3},
        {"unclip_ratio": 3.0},
    ],
)
def test_predict_params(
    monkeypatch,
    seal_text_detection_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        seal_text_detection_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
