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

from typing import Any

import pytest

from langchain_paddleocr import PaddleOCRVLLoader


def test_snake_to_camel_conversion_and_additional_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_service_params: dict[str, Any] = {}

    original_init = PaddleOCRVLLoader.__init__

    def _wrapped_init(self: PaddleOCRVLLoader, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        nonlocal captured_service_params
        captured_service_params = getattr(self, "_service_params")

    monkeypatch.setattr(PaddleOCRVLLoader, "__init__", _wrapped_init)

    _ = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        api_url="http://example.com",
        use_doc_orientation_classify=True,
        layout_unclip_ratio=(0.1, 0.9),
        prompt_label="ocr",
        additional_params={"customOption": 1, "anotherFlag": True},
    )

    assert captured_service_params["useDocOrientationClassify"] is True
    assert captured_service_params["layoutUnclipRatio"] == (0.1, 0.9)
    assert captured_service_params["promptLabel"] == "ocr"
    assert captured_service_params["customOption"] == 1
    assert captured_service_params["anotherFlag"] is True
