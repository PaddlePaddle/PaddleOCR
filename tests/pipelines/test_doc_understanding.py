import pytest

from paddleocr import DocUnderstanding
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def ocr_engine() -> DocUnderstanding:
    return DocUnderstanding()


@pytest.mark.resource_intensive
@pytest.mark.parametrize(
    "input",
    [
        {
            "image": str(TEST_DATA_DIR / "medal_table.png"),
            "query": "识别这份表格的内容",
        },
        {
            "image": str(TEST_DATA_DIR / "table.jpg"),
            "query": "识别这份表格的内容",
        },
    ],
)
def test_predict(ocr_engine: DocUnderstanding, input: dict) -> None:
    """
    Test PaddleOCR's doc understanding functionality.

    Args:
        ocr_engine: An instance of `DocUnderstanding`.
        input: Input dict to be processed.
    """
    result = ocr_engine.predict(input)

    check_simple_inference_result(result)
    res = result[0]
    assert res["result"] is not None
    assert isinstance(res["result"], str)
