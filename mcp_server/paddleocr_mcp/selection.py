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

from __future__ import annotations

from typing import Optional

from .providers import InferenceProvider, normalize_provider

DEFAULT_MODEL = "PP-OCRv6"

QIANFAN_SUPPORTED_MODELS = frozenset(
    {
        "PP-StructureV3",
        "PaddleOCR-VL",
    }
)

SUPPORTED_MODELS = frozenset(
    {
        "PP-OCRv5",
        "PP-OCRv5-latin",
        "PP-OCRv6",
        "PP-StructureV3",
        "PaddleOCR-VL",
        "PaddleOCR-VL-1.5",
        "PaddleOCR-VL-1.6",
    }
)

_MODEL_TOOLS: dict[str, str] = {
    "PP-OCRv5": "ocr",
    "PP-OCRv5-latin": "ocr",
    "PP-OCRv6": "ocr",
    "PP-StructureV3": "pp_structurev3",
    "PaddleOCR-VL": "paddleocr_vl",
    "PaddleOCR-VL-1.5": "paddleocr_vl",
    "PaddleOCR-VL-1.6": "paddleocr_vl",
}


def tool_for_model(model: str) -> str:
    """Return the MCP tool name for a validated model."""
    return _MODEL_TOOLS[model]


def resolve_model(model: Optional[str], provider: str) -> str:
    """Validate and normalize the user-facing model name."""
    normalized = (model or DEFAULT_MODEL).strip()
    normalized_provider = normalize_provider(provider)
    if normalized not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(
            f"Unsupported model: {normalized!r}. Supported models: {supported}."
        )

    if (
        normalized_provider is InferenceProvider.QIANFAN
        and normalized not in QIANFAN_SUPPORTED_MODELS
    ):
        supported = ", ".join(sorted(QIANFAN_SUPPORTED_MODELS))
        raise ValueError(
            f"Model {normalized!r} is not supported with qianfan source. "
            f"Supported models: {supported}."
        )

    return normalized
