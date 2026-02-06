import pytest

from paddleocr import TableClassification
from ..testing_utils import TEST_DATA_DIR, check_simple_inference_result
from .image_classification_common import check_result_item_keys


@pytest.fixture(scope="module")
def table_classification_predictor():
    return TableClassification()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_predict(table_classification_predictor, image_path):
    result = table_classification_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    check_result_item_keys(result[0])
