import pytest

from paddleocr import DocVLM
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def doc_vlm_predictor():
    return DocVLM()


@pytest.mark.resource_intensive
@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "medal_table.png",
    ],
)
def test_predict(doc_vlm_predictor, image_path):
    result = doc_vlm_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "result",
    }


@pytest.mark.resource_intensive
@pytest.mark.parametrize(
    "params",
    [
        {},
    ],
)
def test_predict_params(
    monkeypatch,
    doc_vlm_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        doc_vlm_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
