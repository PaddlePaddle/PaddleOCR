import pytest

from paddleocr import TextDetection
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def text_detection_predictor():
    return TextDetection()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_predict(text_detection_predictor, image_path):
    result = text_detection_predictor.predict(str(image_path))

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
    text_detection_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        text_detection_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
