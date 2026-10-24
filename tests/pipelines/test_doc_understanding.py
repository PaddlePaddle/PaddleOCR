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
