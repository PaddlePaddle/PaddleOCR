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

from .base import MCPCapability


class BaseDocParsingCapability(MCPCapability):
    """Base class for document parsing MCP capabilities.

    Provides shared functionality for document parsing pipelines
    (PP-StructureV3, PaddleOCR-VL, etc.).
    """

    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for document parsing pipelines.

        Returns:
            Base defaults plus chart recognition enabled.
        """
        defaults = super().get_default_options()
        defaults["use_chart_recognition"] = True
        return defaults

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
