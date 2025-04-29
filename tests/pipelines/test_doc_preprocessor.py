import pytest

from paddleocr import DocPreprocessor
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def ocr_engine() -> DocPreprocessor:
    return DocPreprocessor()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "book_rot180.jpg",
    ],
)
def test_predict(ocr_engine: DocPreprocessor, image_path: str) -> None:
    """
    Test PaddleOCR's doc preprocessor functionality.

    Args:
        ocr_engine: An instance of `DocPreprocessor`.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    assert res["angle"] in {0, 90, 180, 270, -1}
    assert res["rot_img"] is not None
    assert res["output_img"] is not None


@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
    ],
)
def test_predict_params(
    monkeypatch,
    ocr_engine: DocPreprocessor,
    params: dict,
) -> None:
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        ocr_engine,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )
