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

from typing import Any

from ..types import DocParsingResult, OCRResult, TextLine


def parse_ocr_result(result_data: dict[str, Any]) -> OCRResult:
    """Parse OCR response payload."""
    ocr_results = result_data.get("ocrResults", [])
    all_texts, all_confidences, text_lines = [], [], []

    for ocr_result in ocr_results:
        pruned = ocr_result["prunedResult"]
        texts = pruned["rec_texts"]
        scores = pruned["rec_scores"]
        boxes = pruned["rec_boxes"]

        for i, text in enumerate(texts):
            if text and text.strip():
                conf = scores[i] if i < len(scores) else 0
                all_texts.append(text.strip())
                all_confidences.append(conf)
                text_lines.append(
                    TextLine(
                        text=text.strip(), confidence=round(conf, 3), bbox=boxes[i]
                    )
                )

    return OCRResult(
        text="\n".join(all_texts),
        confidence=(
            sum(all_confidences) / len(all_confidences) if all_confidences else 0
        ),
        text_lines=text_lines,
    )


def parse_doc_parsing_result(result_data: dict[str, Any]) -> DocParsingResult:
    """Parse document parsing response payload."""
    doc_parsing_results = result_data.get("layoutParsingResults", [])
    markdown_parts = []
    all_images_mapping = {}

    for res in doc_parsing_results:
        markdown_parts.append(res["markdown"]["text"])
        images = res["markdown"]["images"]
        all_images_mapping.update(images)

    return DocParsingResult(
        markdown="\n".join(markdown_parts),
        pages=len(doc_parsing_results),
        images_mapping=all_images_mapping,
    )
