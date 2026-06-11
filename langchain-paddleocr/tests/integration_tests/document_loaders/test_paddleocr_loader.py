from __future__ import annotations

from pathlib import Path

import pytest

from langchain_paddleocr.document_loaders.paddleocr import (
    PaddleOCRConfig,
    PaddleOCRLoader,
)


@pytest.mark.requires("paddleocr")
def test_paddleocr_loader_ocr_mode_on_image() -> None:
    """Integration test: basic OCR on a sample image.

    Requires a working PaddleOCR installation with downloaded models.
    """
    tests_dir = Path(__file__).resolve().parents[2]
    sample_image = tests_dir / "data" / "sample_img.jpg"

    if not sample_image.exists():
        pytest.skip(f"Sample image not found: {sample_image}")

    config = PaddleOCRConfig(lang="en")
    loader = PaddleOCRLoader(file_path=str(sample_image), config=config)
    docs = list(loader.lazy_load())

    assert len(docs) >= 1
    for doc in docs:
        assert isinstance(doc.page_content, str)
        assert doc.metadata["source"] == str(sample_image)
        assert doc.metadata["engine"] == "paddleocr"
        assert "page" in doc.metadata
        assert "confidence" in doc.metadata


@pytest.mark.requires("paddleocr")
def test_paddleocr_loader_ocr_mode_on_pdf() -> None:
    """Integration test: basic OCR on a sample PDF."""
    tests_dir = Path(__file__).resolve().parents[2]
    sample_pdf = tests_dir / "data" / "sample_pdf.pdf"

    if not sample_pdf.exists():
        pytest.skip(f"Sample PDF not found: {sample_pdf}")

    config = PaddleOCRConfig(lang="en")
    loader = PaddleOCRLoader(file_path=str(sample_pdf), config=config)
    docs = list(loader.lazy_load())

    assert len(docs) >= 1
    for doc in docs:
        assert isinstance(doc.page_content, str)
        assert doc.metadata["source"] == str(sample_pdf)
        assert doc.metadata["engine"] == "paddleocr"


@pytest.mark.requires("paddleocr")
def test_paddleocr_loader_structure_mode_on_image() -> None:
    """Integration test: structure mode on a sample image."""
    tests_dir = Path(__file__).resolve().parents[2]
    sample_image = tests_dir / "data" / "sample_img.jpg"

    if not sample_image.exists():
        pytest.skip(f"Sample image not found: {sample_image}")

    config = PaddleOCRConfig(lang="en")
    loader = PaddleOCRLoader(
        file_path=str(sample_image),
        use_structure=True,
        config=config,
    )
    docs = list(loader.lazy_load())

    assert len(docs) >= 1
    for doc in docs:
        assert isinstance(doc.page_content, str)
        assert doc.metadata["engine"] == "ppstructurev3"
        assert "layout_blocks" in doc.metadata
