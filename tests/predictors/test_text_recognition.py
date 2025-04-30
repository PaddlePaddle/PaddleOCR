import pytest

from paddleocr import TextRecognition
from ..testing_utils import TEST_DATA_DIR, check_simple_inference_result


@pytest.fixture(scope="module")
def text_recognition_predictor():
    return TextRecognition()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "textline.png",
    ],
)
def test_predict(text_recognition_predictor, image_path):
    result = text_recognition_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "rec_text",
        "rec_score",
    }
