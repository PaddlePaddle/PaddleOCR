import pytest

from paddleocr import TextLineOrientationClassification
from ..testing_utils import TEST_DATA_DIR, check_simple_inference_result
from .image_classification_common import check_result_item_keys


@pytest.fixture(scope="module")
def text_line_orientation_classification_predictor():
    return TextLineOrientationClassification()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "textline_rot180.jpg",
    ],
)
def test_predict(text_line_orientation_classification_predictor, image_path):
    result = text_line_orientation_classification_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    check_result_item_keys(result[0])
