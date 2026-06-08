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

from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "PP-OCRv5"

QIANFAN_SUPPORTED_MODELS = frozenset(
    {
        "PP-StructureV3",
        "PaddleOCR-VL",
    }
)

SUPPORTED_MODELS = frozenset(
    {
        "PP-OCRv5",
        "PP-OCRv6",
        "PP-StructureV3",
        "PaddleOCR-VL",
        "PaddleOCR-VL-1.5",
        "PaddleOCR-VL-1.6",
    }
)


@dataclass(frozen=True)
class ResolvedModel:
    """Normalized model selection for MCP startup."""

    model: str
    tool: str
    pipeline: str
    ocr_version: Optional[str] = None
    vl_version: Optional[str] = None


_MODEL_SPECS: dict[str, dict[str, str]] = {
    "PP-OCRv5": {
        "tool": "ocr",
        "pipeline": "OCR",
        "ocr_version": "PP-OCRv5",
    },
    "PP-OCRv6": {
        "tool": "ocr",
        "pipeline": "OCR",
        "ocr_version": "PP-OCRv6",
    },
    "PP-StructureV3": {
        "tool": "pp_structurev3",
        "pipeline": "PP-StructureV3",
    },
    "PaddleOCR-VL": {
        "tool": "paddleocr_vl",
        "pipeline": "PaddleOCR-VL",
        "vl_version": "v1",
    },
    "PaddleOCR-VL-1.5": {
        "tool": "paddleocr_vl",
        "pipeline": "PaddleOCR-VL-1.5",
        "vl_version": "v1.5",
    },
    "PaddleOCR-VL-1.6": {
        "tool": "paddleocr_vl",
        "pipeline": "PaddleOCR-VL-1.6",
        "vl_version": "v1.6",
    },
}


def resolve_model(model: Optional[str], source: str) -> ResolvedModel:
    """Resolve user-facing model name into MCP tool and internal pipeline."""
    normalized = (model or DEFAULT_MODEL).strip()
    if normalized not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(
            f"Unsupported model: {normalized!r}. Supported models: {supported}."
        )

    if source == "qianfan" and normalized not in QIANFAN_SUPPORTED_MODELS:
        supported = ", ".join(sorted(QIANFAN_SUPPORTED_MODELS))
        raise ValueError(
            f"Model {normalized!r} is not supported with qianfan source. "
            f"Supported models: {supported}."
        )

    spec = _MODEL_SPECS[normalized]
    return ResolvedModel(
        model=normalized,
        tool=spec["tool"],
        pipeline=spec["pipeline"],
        ocr_version=spec.get("ocr_version"),
        vl_version=spec.get("vl_version"),
    )
