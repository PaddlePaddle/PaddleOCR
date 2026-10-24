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

from __future__ import annotations

from paddleocr import PaddleOCRVLOptions

from langchain_paddleocr import PaddleOCRVLLoader


def test_vl_options_are_mapped_to_paddleocr_vl_options() -> None:
    loader = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        use_doc_orientation_classify=True,
        layout_unclip_ratio=(0.1, 0.9),
        prompt_label="ocr",
        use_ocr_for_image_block=True,
        format_block_content=True,
        markdown_ignore_labels=["image"],
        vlm_extra_args={"temperature": 0.1},
        return_markdown_images=False,
        output_formats=["docx"],
        extra_options={"futureOption": "enabled"},
    )

    assert isinstance(loader._options, PaddleOCRVLOptions)
    assert loader._options.use_doc_orientation_classify is True
    assert loader._options.layout_unclip_ratio == (0.1, 0.9)
    assert loader._options.prompt_label == "ocr"
    assert loader._options.use_ocr_for_image_block is True
    assert loader._options.format_block_content is True
    assert loader._options.markdown_ignore_labels == ["image"]
    assert loader._options.vlm_extra_args == {"temperature": 0.1}
    assert loader._options.return_markdown_images is False
    assert loader._options.output_formats == ["docx"]
    assert loader._options.extra_options == {"futureOption": "enabled"}
    assert loader._options.to_payload()["futureOption"] == "enabled"
