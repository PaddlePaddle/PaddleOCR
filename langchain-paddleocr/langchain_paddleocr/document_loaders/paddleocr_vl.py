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

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from paddleocr import Model, PaddleOCRClient, PaddleOCRVLOptions
from pydantic import SecretStr

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com"

_PAGES_DELIMITER = "\n\f"


def _resolve_vl_model(model: str | Model) -> Model:
    if isinstance(model, Model):
        resolved = model
    else:
        try:
            resolved = Model(model)
        except ValueError as exc:
            msg = f"Unsupported model: {model!r}"
            raise ValueError(msg) from exc

    return resolved


class PaddleOCRVLLoader(BaseLoader):
    """Load documents using the PaddleOCR-VL document parsing API via SDK."""

    def __init__(
        self,
        file_path: str | Iterable[str],
        *,
        access_token: SecretStr | None = None,
        base_url: str | None = None,
        model: str | Model | None = None,
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
        vlm_extra_args: dict[str, Any] | None = None,
        merge_layout_blocks: bool | None = None,
        markdown_ignore_labels: list[str] | None = None,
        prettify_markdown: bool | None = None,
        show_formula_number: bool | None = None,
        restructure_pages: bool | None = None,
        merge_tables: bool | None = None,
        relevel_titles: bool | None = None,
        return_markdown_images: bool | None = None,
        output_formats: list[str] | None = None,
        visualize: bool | None = None,
        extra_options: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> None:
        """Initialize the PaddleOCR-VL loader.

        Args:
            file_path: Single path/URL or an iterable of paths/URLs to PDF or
                image files.
            access_token: Optional access token as ``SecretStr``. If ``None``, the value
                from the ``PADDLEOCR_ACCESS_TOKEN`` environment variable will be used.
            base_url: Base URL of the PaddleOCR API service. Defaults to the official
                service.
            model: PaddleOCR-VL model name or ``Model`` enum value.
            use_doc_orientation_classify: Whether to enable document orientation
                classification.
            use_doc_unwarping: Whether to enable document unwarping.
            use_layout_detection: Whether to enable layout detection.
            use_chart_recognition: Whether to enable chart recognition.
            use_seal_recognition: Whether to enable seal recognition.
            use_ocr_for_image_block: Whether to use OCR for image layout blocks.
            layout_threshold: Layout detection threshold.
            layout_nms: Whether to apply non-maximum suppression for layout detection.
            layout_unclip_ratio: Layout unclip ratio.
            layout_merge_bboxes_mode: Mode for merging layout bounding boxes.
            layout_shape_mode: Layout shape mode.
            prompt_label: Prompt label for the VLM.
            format_block_content: Whether to save formatted Markdown in JSON.
            repetition_penalty: Repetition penalty for VLM sampling.
            temperature: Temperature for VLM sampling.
            top_p: Top-p sampling value for VLM.
            min_pixels: Minimum number of pixels allowed in preprocessing.
            max_pixels: Maximum number of pixels allowed in preprocessing.
            max_new_tokens: Maximum number of tokens generated by the VLM.
            vlm_extra_args: Additional VLM arguments passed through to the service.
            merge_layout_blocks: Whether to merge layout blocks across columns.
            markdown_ignore_labels: Layout labels to ignore in Markdown.
            prettify_markdown: Whether to prettify the Markdown output.
            show_formula_number: Whether to include formula numbers in Markdown.
            restructure_pages: Whether to restructure results across pages.
            merge_tables: Whether to merge tables across pages.
            relevel_titles: Whether to relevel titles.
            return_markdown_images: Whether to return images referenced by Markdown.
            output_formats: Additional export formats such as ``docx``.
            visualize: Whether to include visualization results.
            extra_options: Additional service options passed through as-is.
            timeout: Poll timeout in seconds.
        """
        self._file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        if access_token is None:
            env_value = os.getenv("PADDLEOCR_ACCESS_TOKEN")
            self._token = env_value
        else:
            self._token = access_token.get_secret_value()

        self._base_url = base_url or _DEFAULT_BASE_URL
        self._timeout = timeout

        self._model = _resolve_vl_model(model) if model is not None else None

        options_kwargs: dict[str, Any] = {
            key: val
            for key, val in {
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
                "vlm_extra_args": vlm_extra_args,
                "merge_layout_blocks": merge_layout_blocks,
                "markdown_ignore_labels": markdown_ignore_labels,
                "prettify_markdown": prettify_markdown,
                "show_formula_number": show_formula_number,
                "restructure_pages": restructure_pages,
                "merge_tables": merge_tables,
                "relevel_titles": relevel_titles,
                "return_markdown_images": return_markdown_images,
                "output_formats": output_formats,
                "visualize": visualize,
                "extra_options": extra_options,
            }.items()
            if val is not None
        }

        self._options = PaddleOCRVLOptions(**options_kwargs) if options_kwargs else None

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        return path.startswith(("http://", "https://"))

    def _process_file(self, file_path: str) -> tuple[str, dict[str, Any]]:
        """Process a single file through the SDK and return text + raw result."""
        with PaddleOCRClient(
            token=self._token,
            base_url=self._base_url,
            client_platform="langchain",
            poll_timeout=self._timeout,
        ) as client:
            parse_kwargs: dict[str, Any] = {"options": self._options}
            if self._model is not None:
                parse_kwargs["model"] = self._model
            if self._is_url(file_path):
                result = client.parse_document(file_url=file_path, **parse_kwargs)
            else:
                local_path = Path(file_path)
                if not local_path.exists():
                    msg = f"File not found: '{file_path}'"
                    raise ValueError(msg)
                result = client.parse_document(
                    file_path=str(local_path),
                    **parse_kwargs,
                )

        text_parts = [page.markdown_text for page in result.pages if page.markdown_text]
        text = _PAGES_DELIMITER.join(text_parts)

        raw_response = {
            "job_id": result.job_id,
            "data_info": result.data_info,
            "pages": [
                {
                    "markdown_text": page.markdown_text,
                    "markdown_images": page.markdown_images,
                    "output_images": page.output_images,
                    "pruned_result": page.pruned_result,
                    "input_image_url": page.input_image_url,
                    "exports": page.exports,
                    "markdown": page.markdown,
                    "raw": page.raw,
                }
                for page in result.pages
            ],
        }

        return text, raw_response

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the configured file paths."""
        for file_path in self._file_paths:
            text, raw_response = self._process_file(file_path)

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
            if self._model is not None:
                metadata["model"] = self._model.value

            yield Document(page_content=text, metadata=metadata)
