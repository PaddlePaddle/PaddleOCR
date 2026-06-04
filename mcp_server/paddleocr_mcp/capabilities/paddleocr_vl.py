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

from typing import Any, Dict, List, Optional, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from .doc_parsing_base import BaseDocParsingCapability


class PaddleOCRVLCapability(BaseDocParsingCapability):
    """PaddleOCR-VL series document parsing MCP capability.

    Handles PaddleOCR-VL, PaddleOCR-VL-1.5, and PaddleOCR-VL-1.6.
    """

    PIPELINE: str

    def __init__(self, executor, pipeline: str = "PaddleOCR-VL"):
        """Initialize PaddleOCR-VL capability.

        Args:
            executor: Executor instance.
            pipeline: Specific PaddleOCR-VL version (e.g., "PaddleOCR-VL-1.5").
        """
        if pipeline not in ("PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"):
            raise ValueError(f"Unknown PaddleOCR-VL pipeline: {pipeline}")
        super().__init__(executor)
        self.PIPELINE = pipeline
        self._tool_name = "paddleocr_vl"

    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for PaddleOCR-VL series.

        Returns:
            Base doc parsing defaults plus seal recognition enabled.
        """
        defaults = super().get_default_options()
        defaults["use_seal_recognition"] = True
        return defaults

    def register_tools(self, mcp: Any) -> None:
        @mcp.tool(self._tool_name)
        async def _doc_parsing(
            input_data: str,
            output_mode: str = "simple",
            file_type: Optional[str] = None,
            return_images: bool = True,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """Extract structured Markdown from complex documents.

            Args:
                input_data: File path, URL, or Base64 string.
                output_mode: Output mode.
                    - "simple": Clear readable Markdown (default).
                    - "detailed": JSON format containing document structure.
                file_type: File type (required for URL).
                return_images: Whether to return extracted images.
            """
            return await self._process(
                input_data, output_mode, ctx, file_type, return_images=return_images
            )
