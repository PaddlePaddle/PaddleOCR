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

from paddleocr import TextImageUnwarping
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def text_image_unwarping_predictor():
    return TextImageUnwarping()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "book.jpg",
    ],
)
def test_predict(text_image_unwarping_predictor, image_path):
    result = text_image_unwarping_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    assert result[0].keys() == {
        "input_path",
        "page_index",
        "input_img",
        "doctr_img",
    }


@pytest.mark.parametrize(
    "params",
    [
        {},
    ],
)
def test_predict_params(
    monkeypatch,
    text_image_unwarping_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        text_image_unwarping_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
