from __future__ import annotations

import base64
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

# Bypass model source check to save import time
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "false"

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from paddlex.inference.serving.schemas.paddleocr_vl import (
    InferRequest as PaddleOCRVLInferRequest,
)
from paddlex.inference.serving.schemas.paddleocr_vl import (
    InferResult as PaddleOCRVLInferResult,
)
from paddlex.inference.serving.schemas.shared.ocr import (
    FileType,
)
from pydantic import SecretStr

logger = logging.getLogger(__name__)


FileTypeInput = Literal["pdf", "image"] | None

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

_PAGES_DELIMITER = "\n\f"


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    if not name:
        return name
    parts = name.split("_")
    if len(parts) == 1:
        return name
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _snake_keys_to_camel(params: dict[str, Any]) -> dict[str, Any]:
    """Convert a dict with snake_case keys to camelCase, keeping only non-None
    values."""
    return {
        _snake_to_camel(key): value
        for key, value in params.items()
        if value is not None
    }


def _normalize_file_type(file_type: FileTypeInput) -> FileType | None:
    """Normalize file type to the numeric format required by the API."""
    if file_type is None:
        return None
    if isinstance(file_type, str):
        lower = file_type.lower()
        if lower == "pdf":
            return 0
        if lower == "image":
            return 1
        msg = f"Invalid `file_type` string: {file_type}. Must be 'pdf' or 'image'."
        raise ValueError(msg)
    msg = f"Invalid `file_type` value: {file_type}. Must be 'pdf', 'image', or `None`."
    raise ValueError(msg)


def _infer_file_type_from_path(path: Path) -> FileType | None:
    """
    Infer file type (PDF or image) from file extension.

    Returns:
        0 for PDF, 1 for image, or None if the type cannot be determined.
    """
    suffix = path.suffix.lower()
    if suffix in _PDF_EXTENSIONS:
        return 0
    if suffix in _IMAGE_EXTENSIONS:
        return 1
    return None


class PaddleOCRVLLoader(BaseLoader):
    """Load documents using the PaddleOCR-VL document parsing API."""

    def __init__(
        self,
        file_path: str | Iterable[str],
        *,
        api_url: str,
        access_token: SecretStr | None = None,
        file_type: FileTypeInput = None,
        use_doc_orientation_classify: bool | None = False,
        use_doc_unwarping: bool | None = False,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        use_seal_recognition: bool | None = None,
        use_ocr_for_image_block: bool | None = None,
        layout_threshold: float | dict[int, float] | None = None,
        layout_nms: bool | None = None,
        layout_unclip_ratio: (
            tuple[float, float] | dict[int, tuple[float, float]] | float | None
        ) = None,
        layout_merge_bboxes_mode: str | dict[str, float] | None = None,
        layout_shape_mode: str | None = None,
        prompt_label: str | None = None,
        format_block_content: bool | None = None,
        repetition_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int | None = None,
        merge_layout_blocks: bool | None = None,
        markdown_ignore_labels: list[str] | None = None,
        vlm_extra_args: dict[str, Any] | None = None,
        prettify_markdown: bool | None = None,
        show_formula_number: bool | None = None,
        restructure_pages: bool | None = None,
        merge_tables: bool | None = None,
        relevel_titles: bool | None = None,
        visualize: bool | None = None,
        additional_params: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> None:
        """Initialize the PaddleOCR-VL loader.

        Args:
            file_path: Single path/URL or an iterable of paths/URLs to PDF or
                image files.
            api_url: URL of the PaddleOCR-VL API endpoint.
            access_token: Optional access token as ``SecretStr``. If ``None``, the value
                from the ``PADDLEOCR_ACCESS_TOKEN`` environment variable will be used,
                if present.
            file_type: File type hint. ``"pdf"`` for PDFs, ``"image"`` for images,
                or ``None`` to infer from the file extension.
            use_doc_orientation_classify: Whether to enable document orientation
                classification.
            use_doc_unwarping: Whether to enable document unwarping.
            use_layout_detection: Whether to enable layout detection.
            use_chart_recognition: Whether to enable chart recognition.
            use_seal_recognition: Whether to enable seal recognition.
            use_ocr_for_image_block: Whether to run OCR on image blocks.
            layout_threshold: Layout detection threshold (float or page-specific dict).
            layout_nms: Whether to apply non-maximum suppression for layout detection.
            layout_unclip_ratio: Layout unclip ratio (float, (min, max) tuple, or dict).
            layout_merge_bboxes_mode: Mode for merging layout bounding boxes.
            layout_shape_mode: Layout shape mode.
            prompt_label: Prompt label for the VLM (for example, ``"ocr"`` or
                ``"table"``).
            format_block_content: Whether to format block content.
            repetition_penalty: Repetition penalty for VLM sampling.
            temperature: Temperature for VLM sampling.
            top_p: Top-p sampling value for VLM.
            min_pixels: Minimum number of pixels allowed in preprocessing.
            max_pixels: Maximum number of pixels allowed in preprocessing.
            max_new_tokens: Maximum number of tokens generated by the VLM.
            merge_layout_blocks: Whether to merge layout blocks across columns.
            markdown_ignore_labels: Layout labels to ignore when generating Markdown.
            vlm_extra_args: Additional configuration parameters for the VLM.
            prettify_markdown: Whether to prettify the Markdown output.
            show_formula_number: Whether to include formula numbers in Markdown.
            restructure_pages: Whether to restructure results across pages.
            merge_tables: Whether to merge tables across pages.
            relevel_titles: Whether to relevel titles.
            visualize: Whether to include visualization results.
            additional_params: Additional parameters to pass directly to the API
                (keys are treated as snake_case and converted to camelCase).
            timeout: Request timeout in seconds.
        """
        self._file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        self.api_url = api_url
        self.timeout = timeout

        if access_token is None:
            env_value = os.getenv("PADDLEOCR_ACCESS_TOKEN")
            self.access_token = SecretStr(env_value) if env_value else None
        else:
            self.access_token = access_token

        self.file_type = _normalize_file_type(file_type)

        base_params: dict[str, Any] = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_layout_detection": use_layout_detection,
            "use_chart_recognition": use_chart_recognition,
            "use_seal_recognition": use_seal_recognition,
            "use_ocr_for_image_block": use_ocr_for_image_block,
            "layout_threshold": layout_threshold,
            "layout_nms": layout_nms,
            "layout_unclip_ratio": layout_unclip_ratio,
            "layout_merge_bboxes_mode": layout_merge_bboxes_mode,
            "layout_shape_mode": layout_shape_mode,
            "prompt_label": prompt_label,
            "format_block_content": format_block_content,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_new_tokens": max_new_tokens,
            "merge_layout_blocks": merge_layout_blocks,
            "markdown_ignore_labels": markdown_ignore_labels,
            "vlm_extra_args": vlm_extra_args,
            "prettify_markdown": prettify_markdown,
            "show_formula_number": show_formula_number,
            "restructure_pages": restructure_pages,
            "merge_tables": merge_tables,
            "relevel_titles": relevel_titles,
            "visualize": visualize,
        }

        # Convert all known service parameters from snake_case to camelCase.
        service_params = _snake_keys_to_camel(base_params)

        if additional_params:
            for key, value in additional_params.items():
                if value is None:
                    continue
                service_params[key] = value

        self._service_params = service_params

    def _read_file_bytes(self, path: Path) -> str:
        """Read file content from a local path."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            msg = f"Failed to read file from path '{path}': {exc}"
            raise ValueError(msg) from exc
        return base64.b64encode(data).decode("ascii")

    def _call_api(self, data: str, file_type: FileType) -> tuple[str, dict[str, Any]]:
        """Call the PaddleOCR-VL API and return extracted text and raw response."""

        request_data: dict[str, Any] = {
            "file": data,
            "fileType": file_type,
        }
        request_data.update(self._service_params)

        try:
            request_model = PaddleOCRVLInferRequest(**request_data)
            request_payload = request_model.model_dump(exclude_none=True)
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Invalid request parameters for PaddleOCR-VL: {exc}"
            raise ValueError(msg) from exc

        headers = {
            "Content-Type": "application/json",
            "Client-Platform": "langchain",
        }
        if self.access_token is not None:
            headers["Authorization"] = f"token {self.access_token.get_secret_value()}"

        try:
            response = requests.post(
                self.api_url,
                json=request_payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            msg = f"Failed to call PaddleOCR-VL API: {exc}"
            raise ValueError(msg) from exc

        try:
            response_data: dict[str, Any] = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            msg = f"Invalid JSON response from PaddleOCR-VL API: {exc}"
            raise ValueError(msg) from exc

        if "result" not in response_data:
            msg = "Response from PaddleOCR-VL API is missing the 'result' field."
            raise ValueError(msg)

        try:
            result = PaddleOCRVLInferResult(**response_data["result"])
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Invalid response format from PaddleOCR-VL API: {exc}"
            raise ValueError(msg) from exc

        text_parts = [
            layout_result.markdown.text
            for layout_result in result.layoutParsingResults
            if layout_result.markdown and layout_result.markdown.text
        ]
        text = _PAGES_DELIMITER.join(text_parts)

        return text, response_data

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the configured file paths."""
        for file_path in self._file_paths:
            if self.file_type is not None:
                file_type = self.file_type
            else:
                file_type = None

            if file_path.startswith(("http://", "https://")):
                data = file_path
            else:
                data = self._read_file_bytes(Path(file_path))
                if file_type is None:
                    inferred = _infer_file_type_from_path(Path(file_path))
                    if inferred is None:
                        msg = f"Could not determine file type for '{file_path}'."
                        raise ValueError(msg)
                    file_type = inferred

            text, raw_response = self._call_api(data, file_type)

            if not text:
                logger.warning(
                    "%s could not extract any text from '%s'."
                    " Returning an empty document.",
                    self.__class__.__name__,
                    file_path,
                )

            metadata: dict[str, Any] = {
                "source": str(file_path),
                "paddleocr_vl_raw_response": raw_response,
            }

            yield Document(page_content=text, metadata=metadata)
