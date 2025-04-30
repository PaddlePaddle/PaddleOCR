import pytest

from paddleocr import DocImgOrientationClassification
from ..testing_utils import TEST_DATA_DIR, check_simple_inference_result
from .image_classification_common import check_result_item_keys


@pytest.fixture(scope="module")
def doc_img_orientation_classification_predictor():
    return DocImgOrientationClassification()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "book_rot180.jpg",
    ],
)
def test_predict(doc_img_orientation_classification_predictor, image_path):
    result = doc_img_orientation_classification_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    check_result_item_keys(result[0])
