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
from typing import Any, Dict, List, Optional, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from ..executors.base import Executor
from .base import MCPCapability


class OCRCapability(MCPCapability):
    """OCR MCP capability."""

    PIPELINE = "OCR"

    def register_tools(self, mcp: Any) -> None:
        @mcp.tool("ocr")
        async def _ocr(
            input_data: str,
            output_mode: str = "simple",
            file_type: Optional[str] = None,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """Extract text from images and PDFs.

            Args:
                input_data: File path, URL, or Base64 string.
                output_mode: Output mode.
                    - "simple": Clear readable text (default).
                    - "detailed": JSON containing text, confidence, and bounding box coordinates.
                file_type: File type (required for URL).
                    - "image": Image file.
                    - "pdf": PDF document.
            """
            await ctx.info(f"--- OCR tool received `input_data`: {input_data[:50]} ---")
            return await self._process(input_data, output_mode, ctx, file_type)

    def _format_result(
        self, result: Dict[str, Any], detailed: bool, **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["text"].strip():
            return (
                "❌ No text detected"
                if not detailed
                else json.dumps({"error": "No text detected"}, ensure_ascii=False)
            )

        if detailed:
            return json.dumps(result, ensure_ascii=False, indent=2)

        confidence = result["confidence"]
        text_line_count = len(result["text_lines"])

        output = result["text"]
        if confidence > 0:
            output += f"\n\n📊 Confidence: {(confidence * 100):.1f}% | {text_line_count} text lines"

        return output
