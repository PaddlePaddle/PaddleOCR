from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from langchain_paddleocr import PaddleOCRVLLoader


def test_paddleocr_vl_loader_live_integration() -> None:
    """Live integration test against a real PaddleOCR-VL endpoint.

    This test requires the following environment variables to be set:

    - ``PADDLEOCR_VL_API_URL``: The PaddleOCR-VL HTTP endpoint.
    - ``PADDLEOCR_ACCESS_TOKEN``: Access token for the endpoint.
    """
    api_url = os.getenv("PADDLEOCR_VL_API_URL")
    access_token = os.getenv("PADDLEOCR_ACCESS_TOKEN")

    if not api_url or not access_token:
        pytest.skip(
            "PADDLEOCR_VL_API_URL and PADDLEOCR_ACCESS_TOKEN must be"
            " set for integration tests."
        )

    tests_dir = Path(__file__).resolve().parents[2]
    sample_paths = [
        str(tests_dir / "data" / "sample_pdf.pdf"),
        str(tests_dir / "data" / "sample_img.jpg"),
    ]

    loader = PaddleOCRVLLoader(
        file_path=sample_paths,
        api_url=api_url,
        access_token=SecretStr(access_token),
    )

    docs = list(loader.lazy_load())
    assert len(docs) == 2

    for doc, input_path in zip(docs, sample_paths):
        assert isinstance(doc.page_content, str)
        assert doc.metadata.get("source") == input_path
        assert "paddleocr_vl_raw_response" in doc.metadata
