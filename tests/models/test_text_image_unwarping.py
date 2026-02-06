import pytest

from paddleocr import TextImageUnwarping
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def text_image_unwarping_predictor():
    return TextImageUnwarping()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "book.jpg",
    ],
)
def test_predict(text_image_unwarping_predictor, image_path):
    result = text_image_unwarping_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "doctr_img",
    }


@pytest.mark.parametrize(
    "params",
    [
        {},
    ],
)
def test_predict_params(
    monkeypatch,
    text_image_unwarping_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        text_image_unwarping_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
