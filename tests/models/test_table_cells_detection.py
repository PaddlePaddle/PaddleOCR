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

pytestmark = pytest.mark.py38_incompatible

from paddleocr import TableCellsDetection
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)
from .object_detection_common import check_result_item_keys


@pytest.fixture(scope="module")
def table_cells_detection_predictor():
    return TableCellsDetection()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_predict(table_cells_detection_predictor, image_path):
    result = table_cells_detection_predictor.predict(str(image_path))

    check_simple_inference_result(result)
    check_result_item_keys(result[0])


@pytest.mark.parametrize(
    "params",
    [
        {"img_size": 640},
        {"threshold": 0.5},
    ],
)
def test_predict_params(
    monkeypatch,
    table_cells_detection_predictor,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        table_cells_detection_predictor,
        "paddlex_predictor",
        "dummy_path",
        params,
    )
