import pytest

from paddleocr import FormulaRecognition
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def formula_recognition_predictor():
    return FormulaRecognition()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "formula.png",
    ],
)
def test_predict(formula_recognition_predictor, image_path):
    result = formula_recognition_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "rec_formula",
    }


@pytest.mark.parametrize(
    "params",
    [
        {},
    ],
)
def test_predict_params(
    monkeypatch,
    formula_recognition_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        formula_recognition_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
