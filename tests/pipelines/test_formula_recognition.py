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

from paddleocr import FormulaRecognitionPipeline
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def formula_recognition_engine() -> FormulaRecognitionPipeline:
    return FormulaRecognitionPipeline()


# TODO: Should we separate unit tests and integration tests?
@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "doc_with_formula.png",
    ],
)
def test_predict(
    formula_recognition_engine: FormulaRecognitionPipeline, image_path: str
) -> None:
    """
    Test FormulaRecognitionPipeline's formula_recognition functionality.

    Args:
        formula_recognition_engine: An instance of `FormulaRecognitionPipeline`.
        image_path: Path to the image to be processed.
    """
    result = formula_recognition_engine.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    assert isinstance(res["formula_res_list"], list)
    assert len(res["formula_res_list"]) > 0


# TODO: Also check passing `None`
@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_layout_detection": False},
        {"layout_threshold": 0.5},
        {"layout_nms": True},
        {"layout_unclip_ratio": 1.5},
        {"layout_merge_bboxes_mode": "large"},
    ],
)
def test_predict_params(
    monkeypatch,
    formula_recognition_engine: FormulaRecognitionPipeline,
    params: dict,
) -> None:
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        formula_recognition_engine,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test init params
