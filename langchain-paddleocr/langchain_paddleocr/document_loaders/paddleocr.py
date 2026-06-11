"""PaddleOCR document loader for local OCR inference.

This module provides ``PaddleOCRLoader``, a LangChain document loader that wraps
the local PaddleOCR library (PP-OCRv5 and PP-StructureV3) to extract text from
PDF and image files -- without requiring any cloud API.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

_PDF_EXTENSIONS = {".pdf"}

_SUPPORTED_EXTENSIONS = _IMAGE_EXTENSIONS | _PDF_EXTENSIONS

_PAGES_DELIMITER = "\n\f"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PaddleOCRLoaderError(Exception):
    """Base exception for all errors raised by :class:`PaddleOCRLoader`."""


class UnsupportedFileTypeError(PaddleOCRLoaderError):
    """Raised when a file has an unsupported extension."""


class FileReadError(PaddleOCRLoaderError):
    """Raised when a file cannot be read from disk."""


class OCREngineError(PaddleOCRLoaderError):
    """Raised when the PaddleOCR engine fails during inference."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PaddleOCRConfig:
    """Configuration for the PaddleOCR local inference engine.

    All fields mirror the PaddleOCR / PP-StructureV3 constructor parameters.
    Fields left as ``None`` use the library defaults.

    Example::

        config = PaddleOCRConfig(lang="en", use_doc_orientation_classify=True)
        loader = PaddleOCRLoader(file_path="doc.pdf", config=config)
    """

    # -- Language & version -------------------------------------------------
    lang: str | None = None
    """Language code for OCR (e.g. ``"ch"``, ``"en"``, ``"fr"``).
    Defaults to the library default (``"ch"``)."""

    ocr_version: str | None = None
    """OCR pipeline version (``"PP-OCRv3"``, ``"PP-OCRv4"``, ``"PP-OCRv5"``)."""

    # -- Document pre-processing --------------------------------------------
    use_doc_orientation_classify: bool | None = None
    """Enable automatic document orientation classification."""

    use_doc_unwarping: bool | None = None
    """Enable document de-warping (straightening curved text)."""

    use_textline_orientation: bool | None = None
    """Enable text-line orientation correction."""

    # -- Text detection -----------------------------------------------------
    text_detection_model_name: str | None = None
    text_detection_model_dir: str | None = None
    text_det_limit_side_len: int | None = None
    text_det_limit_type: str | None = None
    text_det_thresh: float | None = None
    text_det_box_thresh: float | None = None
    text_det_unclip_ratio: float | None = None

    # -- Text recognition ---------------------------------------------------
    text_recognition_model_name: str | None = None
    text_recognition_model_dir: str | None = None
    text_recognition_batch_size: int | None = None
    text_rec_score_thresh: float | None = None

    # -- Structure mode (PP-StructureV3) ------------------------------------
    use_table_recognition: bool | None = None
    """Enable table structure recognition (structure mode only)."""

    use_formula_recognition: bool | None = None
    """Enable formula recognition (structure mode only)."""

    use_chart_recognition: bool | None = None
    """Enable chart recognition (structure mode only)."""

    use_seal_recognition: bool | None = None
    """Enable seal text recognition (structure mode only)."""

    use_region_detection: bool | None = None
    """Enable region detection (structure mode only)."""

    layout_detection_model_name: str | None = None
    layout_detection_model_dir: str | None = None
    layout_threshold: float | None = None
    layout_nms: bool | None = None
    layout_unclip_ratio: float | None = None
    layout_merge_bboxes_mode: str | None = None

    format_block_content: bool | None = None
    """Whether to format block content in structure mode output."""

    markdown_ignore_labels: list[str] | None = None
    """Layout labels to skip when generating markdown (structure mode only)."""

    def to_engine_kwargs(self) -> dict[str, Any]:
        """Return a dict of non-``None`` fields suitable for engine construction."""
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_file_path(path: Path) -> None:
    """Validate that *path* exists and has a supported extension.

    Raises:
        FileReadError: If the path does not exist or is not a file.
        UnsupportedFileTypeError: If the file extension is not supported.
    """
    if not path.exists():
        msg = f"File not found: '{path}'"
        raise FileReadError(msg)
    if not path.is_file():
        msg = f"Path is not a file: '{path}'"
        raise FileReadError(msg)
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        msg = (
            f"Unsupported file extension '{suffix}' for '{path}'. "
            f"Supported extensions: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
        raise UnsupportedFileTypeError(msg)


def _extract_text_from_ocr_result(result: dict[str, Any]) -> str:
    """Join recognised text lines from a single-page OCR result dict."""
    texts: list[str] = result.get("rec_texts", [])
    return "\n".join(texts)


def _mean_confidence(result: dict[str, Any]) -> float:
    """Compute the mean recognition confidence for one page."""
    scores: list[float] = result.get("rec_scores", [])
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class PaddleOCRLoader(BaseLoader):
    """Load documents using the local PaddleOCR library.

    Supports two modes of operation:

    * **Basic OCR** (default) -- uses :class:`paddleocr.PaddleOCR` to extract
      raw text lines with bounding boxes and confidence scores.
    * **Structure mode** -- uses :class:`paddleocr.PPStructureV3` for
      layout-aware extraction including tables, titles, figures, and formulas.

    Example -- basic OCR::

        from langchain_paddleocr import PaddleOCRLoader

        loader = PaddleOCRLoader(file_path="invoice.pdf")
        docs = loader.load()

    Example -- structure mode with custom config::

        from langchain_paddleocr import PaddleOCRLoader
        from langchain_paddleocr.document_loaders.paddleocr import PaddleOCRConfig

        config = PaddleOCRConfig(lang="en", use_table_recognition=True)
        loader = PaddleOCRLoader(
            file_path=["page1.png", "page2.png"],
            use_structure=True,
            config=config,
        )
        for doc in loader.lazy_load():
            print(doc.page_content)
    """

    def __init__(
        self,
        file_path: str | Iterable[str],
        *,
        use_structure: bool = False,
        config: PaddleOCRConfig | None = None,
    ) -> None:
        """Initialise the loader.

        Args:
            file_path: Single path or iterable of paths to PDF / image files.
            use_structure: If ``True``, use PP-StructureV3 for layout-aware
                extraction. Otherwise use basic PaddleOCR text extraction.
            config: Optional :class:`PaddleOCRConfig` with engine parameters.
                When ``None``, library defaults are used.
        """
        self._file_paths: list[str] = (
            list(file_path)
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )
        self._use_structure = use_structure
        self._config = config or PaddleOCRConfig()

    # -- Engine helpers (lazy-imported) -------------------------------------

    def _build_ocr_engine(self) -> Any:
        """Create a :class:`paddleocr.PaddleOCR` engine instance."""
        try:
            from paddleocr import PaddleOCR  # Lazy import -- slow first load
        except ImportError as exc:
            msg = (
                "The 'paddleocr' package is required for PaddleOCRLoader. "
                "Install it with: pip install paddleocr"
            )
            raise ImportError(msg) from exc

        kwargs = self._config.to_engine_kwargs()
        logger.debug("Initialising PaddleOCR engine with params: %s", kwargs)
        return PaddleOCR(**kwargs)

    def _build_structure_engine(self) -> Any:
        """Create a :class:`paddleocr.PPStructureV3` engine instance."""
        try:
            from paddleocr import PPStructureV3  # Lazy import
        except ImportError as exc:
            msg = (
                "The 'paddleocr' package is required for PaddleOCRLoader "
                "structure mode. Install it with: pip install paddleocr"
            )
            raise ImportError(msg) from exc

        kwargs = self._config.to_engine_kwargs()
        logger.debug(
            "Initialising PPStructureV3 engine with params: %s",
            kwargs,
        )
        return PPStructureV3(**kwargs)

    # -- Core inference -----------------------------------------------------

    def _process_with_ocr(
        self,
        engine: Any,
        file_path: str,
    ) -> Iterator[Document]:
        """Run basic OCR on *file_path* and yield one Document per page."""
        try:
            results: list[dict[str, Any]] = engine.predict(file_path)
        except Exception as exc:
            msg = f"PaddleOCR engine failed on '{file_path}': {exc}"
            raise OCREngineError(msg) from exc

        if not results:
            logger.warning(
                "%s: OCR returned no results for '%s'. Yielding empty document.",
                self.__class__.__name__,
                file_path,
            )
            yield Document(
                page_content="",
                metadata={"source": file_path, "engine": "paddleocr"},
            )
            return

        for page_index, page_result in enumerate(results):
            text = _extract_text_from_ocr_result(page_result)
            confidence = _mean_confidence(page_result)

            if not text:
                logger.warning(
                    "%s: No text extracted from page %d of '%s'.",
                    self.__class__.__name__,
                    page_index,
                    file_path,
                )

            metadata: dict[str, Any] = {
                "source": file_path,
                "page": page_index,
                "confidence": confidence,
                "language": self._config.lang or "ch",
                "engine": "paddleocr",
            }
            yield Document(page_content=text, metadata=metadata)

    def _process_with_structure(
        self,
        engine: Any,
        file_path: str,
    ) -> Iterator[Document]:
        """Run PP-StructureV3 on *file_path* and yield one Document per page."""
        try:
            results: list[dict[str, Any]] = engine.predict(file_path)
        except Exception as exc:
            msg = f"PPStructureV3 engine failed on '{file_path}': {exc}"
            raise OCREngineError(msg) from exc

        if not results:
            logger.warning(
                "%s: Structure analysis returned no results for '%s'. "
                "Yielding empty document.",
                self.__class__.__name__,
                file_path,
            )
            yield Document(
                page_content="",
                metadata={"source": file_path, "engine": "ppstructurev3"},
            )
            return

        for page_index, page_result in enumerate(results):
            # --- Extract text from overall OCR result if available ---------
            overall_ocr: dict[str, Any] = page_result.get("overall_ocr_res", {})
            ocr_text = _extract_text_from_ocr_result(overall_ocr)
            confidence = _mean_confidence(overall_ocr)

            # --- Collect layout blocks ------------------------------------
            layout_blocks: list[dict[str, Any]] = page_result.get("layout_blocks", [])

            if not ocr_text and not layout_blocks:
                logger.warning(
                    "%s: No content extracted from page %d of '%s'.",
                    self.__class__.__name__,
                    page_index,
                    file_path,
                )

            metadata: dict[str, Any] = {
                "source": file_path,
                "page": page_index,
                "confidence": confidence,
                "language": self._config.lang or "ch",
                "engine": "ppstructurev3",
                "layout_blocks": layout_blocks,
            }
            yield Document(page_content=ocr_text, metadata=metadata)

    # -- BaseLoader interface -----------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the configured file paths.

        Yields one :class:`~langchain_core.documents.Document` per page per
        file. Multi-page PDFs produce multiple documents.

        Raises:
            FileReadError: If a file path does not exist or is not a file.
            UnsupportedFileTypeError: If a file has an unsupported extension.
            OCREngineError: If the PaddleOCR engine fails during inference.
        """
        # Validate all paths up-front before expensive engine initialisation.
        resolved_paths: list[Path] = []
        for raw_path in self._file_paths:
            path = Path(raw_path)
            _validate_file_path(path)
            resolved_paths.append(path)

        # Build the engine once for all files.
        engine = (
            self._build_structure_engine()
            if self._use_structure
            else self._build_ocr_engine()
        )

        for path in resolved_paths:
            file_str = str(path)
            logger.info(
                "%s: Processing '%s' (mode=%s).",
                self.__class__.__name__,
                file_str,
                "structure" if self._use_structure else "ocr",
            )
            if self._use_structure:
                yield from self._process_with_structure(engine, file_str)
            else:
                yield from self._process_with_ocr(engine, file_str)
