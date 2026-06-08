# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from paddleocr_mcp.inference.paddleocr_vl.local import _PIPELINE_VERSION_BY_MODEL
from paddleocr_mcp.selection import (
    DEFAULT_MODEL,
    resolve_model,
    tool_for_model,
)


def test_resolve_model_defaults_to_default_model():
    model = resolve_model(None, "local")
    assert model == DEFAULT_MODEL
    assert tool_for_model(model) == "ocr"


def test_resolve_model_pp_ocrv6():
    model = resolve_model("PP-OCRv6", "aistudio")
    assert model == "PP-OCRv6"
    assert tool_for_model(model) == "ocr"


def test_resolve_model_pp_structurev3():
    model = resolve_model("PP-StructureV3", "local")
    assert tool_for_model(model) == "pp_structurev3"


def test_resolve_model_paddleocr_vl_versions():
    model = resolve_model("PaddleOCR-VL-1.5", "local")
    assert model == "PaddleOCR-VL-1.5"
    assert tool_for_model(model) == "paddleocr_vl"


def test_resolve_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_model("unknown-model", "local")


def test_resolve_model_rejects_ocr_on_qianfan():
    with pytest.raises(ValueError, match="not supported with qianfan"):
        resolve_model("PP-OCRv6", "qianfan")


@pytest.mark.parametrize(
    "model",
    ["PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"],
)
def test_resolve_model_rejects_unsupported_vl_on_qianfan(model):
    with pytest.raises(ValueError, match="not supported with qianfan"):
        resolve_model(model, "qianfan")


def test_resolve_model_paddleocr_vl_v1_on_qianfan():
    model = resolve_model("PaddleOCR-VL", "qianfan")
    assert model == "PaddleOCR-VL"
    assert tool_for_model(model) == "paddleocr_vl"


def test_vl_local_pipeline_version_mapping():
    assert _PIPELINE_VERSION_BY_MODEL["PaddleOCR-VL"] == "v1"
    assert _PIPELINE_VERSION_BY_MODEL["PaddleOCR-VL-1.5"] == "v1.5"
    assert _PIPELINE_VERSION_BY_MODEL["PaddleOCR-VL-1.6"] == "v1.6"
