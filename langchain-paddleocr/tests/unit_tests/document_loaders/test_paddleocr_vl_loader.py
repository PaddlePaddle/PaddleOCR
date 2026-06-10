from __future__ import annotations

from typing import Any

import pytest

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


def test_options_none_when_no_params() -> None:
    """Ensure options is None when no OCR params are provided."""
    loader = PaddleOCRVLLoader(file_path="dummy.pdf")
    assert loader._options is None


def test_lazy_load_raises_for_missing_file() -> None:
    """Ensure lazy_load raises when a local file does not exist."""
    loader = PaddleOCRVLLoader(
        file_path="nonexistent-file.pdf",
        access_token=None,
    )
    with pytest.raises(ValueError, match="File not found"):
        list(loader.lazy_load())
