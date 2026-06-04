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

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Union


@dataclass(frozen=True)
class InferenceRequest:
    """Inference request payload."""

    input_data: str
    file_type: Optional[str] = None
    runtime_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextLine:
    """A single text line with its bounding box and confidence."""

    text: str
    confidence: float
    bbox: Any


@dataclass(frozen=True)
class OCRResult:
    """OCR result."""

    text: str
    confidence: float
    text_lines: list[TextLine]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocParsingResult:
    """Document parsing result."""

    markdown: str
    pages: int
    images_mapping: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


InferenceResult = Union[OCRResult, DocParsingResult]
