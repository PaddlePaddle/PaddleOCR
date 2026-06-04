# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import List, Union
from typing_extensions import override

from mcp.types import ImageContent, TextContent

from ..inference.types import InferenceResult, OCRResult
from .base import Task


class OCRTask(Task):
    @property
    @override
    def tool_name(self) -> str:
        return "ocr"

    @override
    def _format_result(
        self, result: InferenceResult, detailed: bool, **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not isinstance(result, OCRResult):
            raise TypeError(f"OCRTask expected OCRResult, got {type(result).__name__}")

        if not result.text.strip():
            return (
                "No text detected"
                if not detailed
                else json.dumps({"error": "No text detected"}, ensure_ascii=False)
            )

        if detailed:
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

        confidence = result.confidence
        text_line_count = len(result.text_lines)

        output = result.text
        if confidence > 0:
            output += f"\n\nConfidence: {(confidence * 100):.1f}% | {text_line_count} text lines"

        return output
