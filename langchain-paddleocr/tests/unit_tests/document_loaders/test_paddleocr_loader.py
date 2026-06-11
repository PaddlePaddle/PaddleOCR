from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_paddleocr.document_loaders.paddleocr import (
    FileReadError,
    OCREngineError,
    PaddleOCRConfig,
    PaddleOCRLoader,
    UnsupportedFileTypeError,
    _extract_text_from_ocr_result,
    _mean_confidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_OCR_PAGE: dict[str, Any] = {
    "rec_texts": ["Hello", "World"],
    "rec_scores": [0.98, 0.95],
    "dt_polys": [
        [[0, 0], [100, 0], [100, 20], [0, 20]],
        [[0, 30], [100, 30], [100, 50], [0, 50]],
    ],
}

SAMPLE_OCR_PAGE_EMPTY: dict[str, Any] = {
    "rec_texts": [],
    "rec_scores": [],
    "dt_polys": [],
}

SAMPLE_STRUCTURE_PAGE: dict[str, Any] = {
    "overall_ocr_res": {
        "rec_texts": ["Title", "Some body text"],
        "rec_scores": [0.99, 0.92],
    },
    "layout_blocks": [
        {"type": "title", "bbox": [0, 0, 200, 40], "res": {}},
        {"type": "text", "bbox": [0, 50, 200, 200], "res": {}},
    ],
}


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    return img


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    return pdf


@pytest.fixture
def tmp_unsupported(tmp_path: Path) -> Path:
    f = tmp_path / "data.xyz"
    f.write_bytes(b"some data")
    return f


# ---------------------------------------------------------------------------
# PaddleOCRConfig tests
# ---------------------------------------------------------------------------


class TestPaddleOCRConfig:
    def test_default_config_produces_empty_kwargs(self) -> None:
        config = PaddleOCRConfig()
        assert config.to_engine_kwargs() == {}

    def test_non_none_fields_are_included(self) -> None:
        config = PaddleOCRConfig(lang="en", text_det_thresh=0.5)
        kwargs = config.to_engine_kwargs()
        assert kwargs == {"lang": "en", "text_det_thresh": 0.5}

    def test_none_fields_are_excluded(self) -> None:
        config = PaddleOCRConfig(lang="fr")
        kwargs = config.to_engine_kwargs()
        assert "ocr_version" not in kwargs
        assert "text_detection_model_dir" not in kwargs


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_text_joins_lines(self) -> None:
        text = _extract_text_from_ocr_result(SAMPLE_OCR_PAGE)
        assert text == "Hello\nWorld"

    def test_extract_text_empty_result(self) -> None:
        assert _extract_text_from_ocr_result(SAMPLE_OCR_PAGE_EMPTY) == ""

    def test_extract_text_missing_key(self) -> None:
        assert _extract_text_from_ocr_result({}) == ""

    def test_mean_confidence_normal(self) -> None:
        conf = _mean_confidence(SAMPLE_OCR_PAGE)
        assert conf == pytest.approx(0.965)

    def test_mean_confidence_empty(self) -> None:
        assert _mean_confidence(SAMPLE_OCR_PAGE_EMPTY) == 0.0

    def test_mean_confidence_missing_key(self) -> None:
        assert _mean_confidence({}) == 0.0


# ---------------------------------------------------------------------------
# File validation tests
# ---------------------------------------------------------------------------


class TestFileValidation:
    def test_nonexistent_file_raises_file_read_error(self) -> None:
        loader = PaddleOCRLoader(file_path="nonexistent.png")
        with pytest.raises(FileReadError, match="File not found"):
            list(loader.lazy_load())

    def test_directory_path_raises_file_read_error(self, tmp_path: Path) -> None:
        loader = PaddleOCRLoader(file_path=str(tmp_path))
        with pytest.raises(FileReadError, match="not a file"):
            list(loader.lazy_load())

    def test_unsupported_extension_raises(self, tmp_unsupported: Path) -> None:
        loader = PaddleOCRLoader(file_path=str(tmp_unsupported))
        with pytest.raises(UnsupportedFileTypeError, match="Unsupported file"):
            list(loader.lazy_load())

    def test_supported_image_extensions(self, tmp_path: Path) -> None:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
            f = tmp_path / f"test{ext}"
            f.write_bytes(b"fake")
            loader = PaddleOCRLoader(file_path=str(f))
            # Should not raise during validation -- will fail at engine build
            # which is separate from validation.
            with patch(
                "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
            ) as mock_build:
                mock_engine = MagicMock()
                mock_engine.predict.return_value = [SAMPLE_OCR_PAGE]
                mock_build.return_value = mock_engine
                docs = list(loader.lazy_load())
                assert len(docs) == 1

    def test_pdf_extension_supported(self, tmp_pdf: Path) -> None:
        loader = PaddleOCRLoader(file_path=str(tmp_pdf))
        with patch(
            "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
        ) as mock_build:
            mock_engine = MagicMock()
            mock_engine.predict.return_value = [SAMPLE_OCR_PAGE]
            mock_build.return_value = mock_engine
            docs = list(loader.lazy_load())
            assert len(docs) == 1


# ---------------------------------------------------------------------------
# Basic OCR mode tests
# ---------------------------------------------------------------------------


class TestOCRMode:
    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_single_page_document(self, mock_build: MagicMock, tmp_image: Path) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [SAMPLE_OCR_PAGE]
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image))
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "Hello\nWorld"
        assert doc.metadata["source"] == str(tmp_image)
        assert doc.metadata["page"] == 0
        assert doc.metadata["confidence"] == pytest.approx(0.965)
        assert doc.metadata["engine"] == "paddleocr"

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_multi_page_pdf(self, mock_build: MagicMock, tmp_pdf: Path) -> None:
        page1 = {"rec_texts": ["Page one"], "rec_scores": [0.90]}
        page2 = {"rec_texts": ["Page two"], "rec_scores": [0.85]}
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [page1, page2]
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_pdf))
        docs = list(loader.lazy_load())

        assert len(docs) == 2
        assert docs[0].page_content == "Page one"
        assert docs[0].metadata["page"] == 0
        assert docs[1].page_content == "Page two"
        assert docs[1].metadata["page"] == 1

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_empty_ocr_result_yields_empty_document(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = []
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image))
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].page_content == ""

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_engine_failure_raises_ocr_engine_error(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.side_effect = RuntimeError("GPU out of memory")
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image))
        with pytest.raises(OCREngineError, match="GPU out of memory"):
            list(loader.lazy_load())

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_multiple_files(
        self, mock_build: MagicMock, tmp_image: Path, tmp_pdf: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [SAMPLE_OCR_PAGE]
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=[str(tmp_image), str(tmp_pdf)])
        docs = list(loader.lazy_load())

        assert len(docs) == 2
        assert docs[0].metadata["source"] == str(tmp_image)
        assert docs[1].metadata["source"] == str(tmp_pdf)

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_custom_config_passed_to_engine(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [SAMPLE_OCR_PAGE]
        mock_build.return_value = mock_engine

        config = PaddleOCRConfig(lang="en", text_det_thresh=0.3)
        loader = PaddleOCRLoader(
            file_path=str(tmp_image),
            config=config,
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].metadata["language"] == "en"

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_ocr_engine"
    )
    def test_default_language_is_ch(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [SAMPLE_OCR_PAGE]
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image))
        docs = list(loader.lazy_load())
        assert docs[0].metadata["language"] == "ch"


# ---------------------------------------------------------------------------
# Structure mode tests
# ---------------------------------------------------------------------------


class TestStructureMode:
    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_structure_engine"
    )
    def test_structure_mode_document(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [SAMPLE_STRUCTURE_PAGE]
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image), use_structure=True)
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "Title\nSome body text"
        assert doc.metadata["engine"] == "ppstructurev3"
        assert len(doc.metadata["layout_blocks"]) == 2
        assert doc.metadata["layout_blocks"][0]["type"] == "title"
        assert doc.metadata["layout_blocks"][1]["type"] == "text"

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_structure_engine"
    )
    def test_structure_mode_empty_result(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.return_value = []
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image), use_structure=True)
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].page_content == ""
        assert docs[0].metadata["engine"] == "ppstructurev3"

    @patch(
        "langchain_paddleocr.document_loaders.paddleocr.PaddleOCRLoader._build_structure_engine"
    )
    def test_structure_engine_failure_raises(
        self, mock_build: MagicMock, tmp_image: Path
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.predict.side_effect = RuntimeError("Model load failed")
        mock_build.return_value = mock_engine

        loader = PaddleOCRLoader(file_path=str(tmp_image), use_structure=True)
        with pytest.raises(OCREngineError, match="Model load failed"):
            list(loader.lazy_load())


# ---------------------------------------------------------------------------
# Constructor / init tests
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_single_string_path_normalised_to_list(self) -> None:
        loader = PaddleOCRLoader(file_path="test.png")
        assert loader._file_paths == ["test.png"]

    def test_iterable_paths_stored(self) -> None:
        loader = PaddleOCRLoader(file_path=["a.png", "b.pdf"])
        assert loader._file_paths == ["a.png", "b.pdf"]

    def test_default_config_created_when_none(self) -> None:
        loader = PaddleOCRLoader(file_path="test.png")
        assert isinstance(loader._config, PaddleOCRConfig)

    def test_use_structure_default_false(self) -> None:
        loader = PaddleOCRLoader(file_path="test.png")
        assert loader._use_structure is False


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


def test_import_from_package() -> None:
    """Verify PaddleOCRLoader is importable from the top-level package."""
    from langchain_paddleocr import PaddleOCRLoader as LoaderAlias

    assert LoaderAlias is PaddleOCRLoader
