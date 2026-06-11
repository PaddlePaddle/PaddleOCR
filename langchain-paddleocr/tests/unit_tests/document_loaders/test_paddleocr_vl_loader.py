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

from unittest.mock import MagicMock

import pytest
from paddleocr import Model

from langchain_paddleocr import PaddleOCRVLLoader


def test_default_base_url() -> None:
    """Ensure default base_url is set when not provided."""
    loader = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        access_token=None,
    )
    assert loader._base_url == "https://paddleocr.aistudio-app.com"


def test_custom_base_url() -> None:
    """Ensure custom base_url is respected."""
    loader = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        base_url="https://custom.example.com",
    )
    assert loader._base_url == "https://custom.example.com"


def test_access_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure token is read from environment variable."""
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "env-token")
    loader = PaddleOCRVLLoader(file_path="dummy.pdf")
    assert loader._token == "env-token"


def test_options_are_built_correctly() -> None:
    """Ensure options with non-None values are passed to PaddleOCRVLOptions."""
    loader = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        use_layout_detection=True,
        temperature=0.7,
        top_p=0.9,
    )
    assert loader._options is not None
    assert loader._options.use_layout_detection is True
    assert loader._options.temperature == 0.7
    assert loader._options.top_p == 0.9


def test_options_include_default_flags() -> None:
    """Ensure default boolean flags are included in options."""
    loader = PaddleOCRVLLoader(file_path="dummy.pdf")
    assert loader._options is not None
    assert loader._options.use_doc_orientation_classify is False
    assert loader._options.use_doc_unwarping is False


def test_lazy_load_raises_for_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure lazy_load raises when a local file does not exist."""
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
    loader = PaddleOCRVLLoader(
        file_path="nonexistent-file.pdf",
    )
    with pytest.raises(ValueError, match="File not found"):
        list(loader.lazy_load())


def test_default_model() -> None:
    loader = PaddleOCRVLLoader(file_path="dummy.pdf")
    assert loader._model is None


def test_parse_document_omits_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_result = MagicMock()
    mock_result.job_id = "job-1"
    mock_result.pages = []

    mock_client = MagicMock()
    mock_client.parse_document.return_value = mock_result
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(
        "langchain_paddleocr.document_loaders.paddleocr_vl.PaddleOCRClient",
        lambda **kwargs: mock_client,
    )
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")

    loader = PaddleOCRVLLoader(file_path="https://example.com/doc.pdf")
    list(loader.lazy_load())

    mock_client.parse_document.assert_called_once_with(
        file_url="https://example.com/doc.pdf",
        options=loader._options,
    )


def test_custom_model_string() -> None:
    loader = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        model="PaddleOCR-VL-1.5",
    )
    assert loader._model == Model.PADDLE_OCR_VL_15


def test_parse_document_receives_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_result = MagicMock()
    mock_result.job_id = "job-1"
    mock_result.pages = []

    mock_client = MagicMock()
    mock_client.parse_document.return_value = mock_result
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(
        "langchain_paddleocr.document_loaders.paddleocr_vl.PaddleOCRClient",
        lambda **kwargs: mock_client,
    )

    loader = PaddleOCRVLLoader(
        file_path="https://example.com/doc.pdf",
        access_token=None,
        model=Model.PADDLE_OCR_VL_15,
    )
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")

    list(loader.lazy_load())

    mock_client.parse_document.assert_called_once_with(
        file_url="https://example.com/doc.pdf",
        model=Model.PADDLE_OCR_VL_15,
        options=loader._options,
    )
