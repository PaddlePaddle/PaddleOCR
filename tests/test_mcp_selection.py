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

from paddleocr_mcp.selection import (
    DEFAULT_MODEL,
    resolve_model,
)


def test_resolve_model_defaults_to_pp_ocrv5():
    resolved = resolve_model(None, "local")
    assert resolved.model == DEFAULT_MODEL
    assert resolved.tool == "ocr"
    assert resolved.pipeline == "OCR"
    assert resolved.ocr_version == "PP-OCRv5"


def test_resolve_model_pp_ocrv6():
    resolved = resolve_model("PP-OCRv6", "aistudio")
    assert resolved.model == "PP-OCRv6"
    assert resolved.tool == "ocr"
    assert resolved.pipeline == "OCR"
    assert resolved.ocr_version == "PP-OCRv6"


def test_resolve_model_pp_structurev3():
    resolved = resolve_model("PP-StructureV3", "local")
    assert resolved.tool == "pp_structurev3"
    assert resolved.pipeline == "PP-StructureV3"


def test_resolve_model_paddleocr_vl_versions():
    resolved = resolve_model("PaddleOCR-VL-1.5", "local")
    assert resolved.tool == "paddleocr_vl"
    assert resolved.pipeline == "PaddleOCR-VL-1.5"
    assert resolved.vl_version == "v1.5"


def test_resolve_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_model("unknown-model", "local")


def test_resolve_model_rejects_ocr_on_qianfan():
    with pytest.raises(ValueError, match="not supported with qianfan"):
        resolve_model("PP-OCRv6", "qianfan")
