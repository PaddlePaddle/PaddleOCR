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

import base64
import io
from typing import Any

from ..types import DocParsingResult


def parse_local_doc_parsing_result(result: Any) -> DocParsingResult:
    markdown_parts: list[str] = []
    all_images_mapping: dict[str, str] = {}

    for res in result:
        markdown = res.markdown
        markdown_parts.append(markdown["markdown_texts"])
        processed_images = {}
        for img_key, img_data in markdown["markdown_images"].items():
            with io.BytesIO() as buffer:
                img_data.save(buffer, format="JPEG")
                processed_images[img_key] = base64.b64encode(buffer.getvalue()).decode(
                    "ascii"
                )
        all_images_mapping.update(processed_images)

    return DocParsingResult(
        markdown="\n".join(markdown_parts),
        pages=len(result),
        images_mapping=all_images_mapping,
    )


def parse_aistudio_doc_parsing_result(result: Any) -> DocParsingResult:
    markdown_parts: list[str] = []
    all_images_mapping: dict[str, str] = {}

    for page in result.pages:
        markdown_parts.append(page.markdown_text)
        all_images_mapping.update(page.markdown_images)

    return DocParsingResult(
        markdown="\n".join(markdown_parts),
        pages=len(result.pages),
        images_mapping=all_images_mapping,
    )
