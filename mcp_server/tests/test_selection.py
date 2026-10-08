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

from __future__ import annotations

import pytest

from paddleocr_mcp.providers import InferenceProvider
from paddleocr_mcp.selection import (
    DEFAULT_MODEL,
    QIANFAN_SUPPORTED_MODELS,
    SUPPORTED_MODELS,
    resolve_model,
    tool_for_model,
)


class TestResolveModel:
    def test_none_falls_back_to_default(self) -> None:
        assert resolve_model(None, "local") == DEFAULT_MODEL

    def test_empty_string_falls_back_to_default(self) -> None:
        assert resolve_model("", "local") == DEFAULT_MODEL

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert resolve_model("  PP-OCRv5  ", "local") == "PP-OCRv5"

    def test_accepts_provider_enum(self) -> None:
        resolved = resolve_model("PP-StructureV3", InferenceProvider.AISTUDIO)
        assert resolved == "PP-StructureV3"

    def test_unsupported_model_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_model("not-a-model", "local")

    def test_qianfan_rejects_model_outside_its_set(self) -> None:
        assert "PP-OCRv5" not in QIANFAN_SUPPORTED_MODELS
        with pytest.raises(ValueError):
            resolve_model("PP-OCRv5", "qianfan")

    def test_qianfan_accepts_supported_model(self) -> None:
        assert resolve_model("PaddleOCR-VL", "qianfan") == "PaddleOCR-VL"


class TestToolForModel:
    def test_ocr_models_map_to_ocr_tool(self) -> None:
        assert tool_for_model("PP-OCRv5") == "ocr"
        assert tool_for_model("PP-OCRv6") == "ocr"

    def test_structure_model_maps_to_structure_tool(self) -> None:
        assert tool_for_model("PP-StructureV3") == "pp_structurev3"

    def test_vl_models_map_to_vl_tool(self) -> None:
        assert tool_for_model("PaddleOCR-VL") == "paddleocr_vl"
        assert tool_for_model("PaddleOCR-VL-1.5") == "paddleocr_vl"


class TestModelRegistries:
    def test_every_supported_model_has_a_tool(self) -> None:
        for model in SUPPORTED_MODELS:
            assert tool_for_model(model)

    def test_default_model_is_supported(self) -> None:
        assert DEFAULT_MODEL in SUPPORTED_MODELS

    def test_qianfan_models_are_supported(self) -> None:
        assert QIANFAN_SUPPORTED_MODELS <= SUPPORTED_MODELS
