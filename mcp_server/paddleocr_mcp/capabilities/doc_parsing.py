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
import re
from typing import Any, Dict, List, Optional, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from ..executors.base import Executor
from .base import MCPCapability


class DocParsingCapability(MCPCapability):
    """Document parsing MCP capability (e.g., PP-StructureV3, PaddleOCR-VL, etc.)."""

    def __init__(self, executor: Executor, tool_name: str):
        """
        Args:
            executor: Executor instance.
            tool_name: MCP tool name (e.g., "pp_structurev3", "paddleocr_vl").
        """
        super().__init__(executor)
        self._tool_name = tool_name

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

    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        return_images: bool = True,
        **kwargs,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["markdown"].strip():
            return (
                "❌ No document content detected"
                if not detailed
                else json.dumps({"error": "No content detected"}, ensure_ascii=False)
            )

        markdown_text = result["markdown"]
        images_mapping = result.get("images_mapping", {})

        if return_images and images_mapping:
            content_list = self._parse_markdown_with_images(
                markdown_text, images_mapping
            )
        else:
            content_list = [TextContent(type="text", text=markdown_text)]

        if detailed:
            content_list.append(
                TextContent(
                    type="text",
                    text=f"Pages: {result['pages']}",
                )
            )

        if len(content_list) == 1:
            return content_list[0]
        return content_list

    def _parse_markdown_with_images(
        self, markdown_text: str, images_mapping: Dict[str, str]
    ) -> List[Union[TextContent, ImageContent]]:
        """Parse markdown text, return mixed text and image content."""
        if not images_mapping:
            return [TextContent(type="text", text=markdown_text)]

        content_list = []
        img_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
        last_pos = 0

        for match in re.finditer(img_pattern, markdown_text):
            text_before = markdown_text[last_pos : match.start()]
            if text_before.strip():
                content_list.append(TextContent(type="text", text=text_before))

            img_src = match.group(1)
            if img_src in images_mapping:
                content_list.append(
                    ImageContent(
                        type="image",
                        data=images_mapping[img_src],
                        mimeType="image/jpeg",
                    )
                )

            last_pos = match.end()

        remaining_text = markdown_text[last_pos:]
        if remaining_text.strip():
            content_list.append(TextContent(type="text", text=remaining_text))

        return content_list or [TextContent(type="text", text=markdown_text)]
