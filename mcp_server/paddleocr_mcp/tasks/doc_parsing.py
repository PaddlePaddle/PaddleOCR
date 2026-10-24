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

import json
import re
from typing import Dict, List, Union
from typing_extensions import override

from mcp.types import ImageContent, TextContent

from ..inference.types import DocParsingResult, InferenceResult
from .base import Task
from .mcp_image import normalize_mcp_image_payload


class DocParsingTask(Task):
    def _format_result(
        self,
        result: InferenceResult,
        detailed: bool,
        return_images: bool = True,
        **kwargs,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not isinstance(result, DocParsingResult):
            raise TypeError(
                f"DocParsingTask expected DocParsingResult, got {type(result).__name__}"
            )

        if not result.markdown.strip():
            return (
                "No document content detected"
                if not detailed
                else json.dumps({"error": "No content detected"}, ensure_ascii=False)
            )

        markdown_text = result.markdown
        images_mapping = result.images_mapping

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
                    text=f"Pages: {result.pages}",
                )
            )

        if len(content_list) == 1:
            return content_list[0]
        return content_list

    def _parse_markdown_with_images(
        self, markdown_text: str, images_mapping: Dict[str, str]
    ) -> List[Union[TextContent, ImageContent]]:
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
                payload = normalize_mcp_image_payload(images_mapping[img_src])
                if payload is not None:
                    image_data, mime_type = payload
                    content_list.append(
                        ImageContent(
                            type="image",
                            data=image_data,
                            mimeType=mime_type,
                        )
                    )
                else:
                    content_list.append(TextContent(type="text", text=match.group(0)))

            last_pos = match.end()

        remaining_text = markdown_text[last_pos:]
        if remaining_text.strip():
            content_list.append(TextContent(type="text", text=remaining_text))

        return content_list or [TextContent(type="text", text=markdown_text)]


class PPStructureV3Task(DocParsingTask):
    @property
    @override
    def tool_name(self) -> str:
        return "pp_structurev3"


class PaddleOCRVLTask(DocParsingTask):
    @property
    @override
    def tool_name(self) -> str:
        return "paddleocr_vl"
