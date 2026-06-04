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

import pytest

from langchain_paddleocr import PaddleOCRVLLoader


def test_file_type_normalization_and_inference(
    tmp_path_factory: pytest.TempdirFactory,
) -> None:
    pdf_file = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.write_bytes(b"%PDF-1.4")

    image_file = tmp_path_factory.mktemp("img") / "sample.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    loader_pdf_hint = PaddleOCRVLLoader(
        file_path=str(pdf_file),
        api_url="http://example.com",
        file_type="pdf",
    )
    assert loader_pdf_hint.file_type == 0

    loader_image_hint = PaddleOCRVLLoader(
        file_path=str(image_file),
        api_url="http://example.com",
        file_type="image",
    )
    assert loader_image_hint.file_type == 1


def test_lazy_load_raises_for_unreadable_file() -> None:
    loader = PaddleOCRVLLoader(
        file_path="nonexistent-file.pdf",
        api_url="http://example.com",
    )
    with pytest.raises(ValueError):
        list(loader.lazy_load())
