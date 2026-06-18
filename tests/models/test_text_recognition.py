# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from paddleocr import TextRecognition
from ..testing_utils import TEST_DATA_DIR, check_simple_inference_result


pytestmark = pytest.mark.py38_incompatible


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
        "vis_font",
    }
