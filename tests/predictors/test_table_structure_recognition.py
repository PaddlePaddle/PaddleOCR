import pytest

from paddleocr import TableStructureRecognition
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def table_structure_recognition_predictor():
    return TableStructureRecognition()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_predict(table_structure_recognition_predictor, image_path):
    result = table_structure_recognition_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "bbox",
        "structure",
        "structure_score",
    }


@pytest.mark.parametrize(
    "params",
    [
        {},
    ],
)
def test_predict_params(
    monkeypatch,
    table_structure_recognition_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        table_structure_recognition_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
