import pytest

from paddleocr import FormulaRecognitionPipeline
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def formula_recognition_engine() -> FormulaRecognitionPipeline:
    return FormulaRecognitionPipeline()


# TODO: Should we separate unit tests and integration tests?
@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "doc_with_formula.png",
    ],
)
def test_predict(
    formula_recognition_engine: FormulaRecognitionPipeline, image_path: str
) -> None:
    """
    Test FormulaRecognitionPipeline's formula_recognition functionality.

    Args:
        formula_recognition_engine: An instance of `FormulaRecognitionPipeline`.
        image_path: Path to the image to be processed.
    """
    result = formula_recognition_engine.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    assert isinstance(res["formula_res_list"], list)
    assert len(res["formula_res_list"]) > 0


# TODO: Also check passing `None`
@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_layout_detection": False},
        {"layout_threshold": 0.5},
        {"layout_nms": True},
        {"layout_unclip_ratio": 1.5},
        {"layout_merge_bboxes_mode": "large"},
    ],
)
def test_predict_params(
    monkeypatch,
    formula_recognition_engine: FormulaRecognitionPipeline,
    params: dict,
) -> None:
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        formula_recognition_engine,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test init params
