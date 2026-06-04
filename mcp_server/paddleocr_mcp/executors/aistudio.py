# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from typing import Any, Dict, Optional

from paddleocr._api_client.async_client import AsyncPaddleOCRClient
from paddleocr._api_client.errors import (
    APIError,
    AuthError,
    JobFailedError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
    ServiceUnavailableError,
)
from paddleocr._api_client.models import (
    Model,
    OCROptions,
    PPStructureV3Options,
    PaddleOCRVLOptions,
)

from .base import (
    AuthenticationError,
    Executor,
    ExecutorError,
    ResourceUnavailableError,
    ExecutionTimeoutError,
)


class AIStudioExecutor(Executor):
    """Executor for AI Studio official API using AsyncPaddleOCRClient.

    Supports OCR, PP-StructureV3, and PaddleOCR-VL series pipelines.
    """

    # Pipeline to SDK Model mapping
    # Maps pipeline names to their corresponding Model enum values from the SDK
    _PIPELINE_MODEL_MAP = {
        "OCR": Model.PP_OCRV5,
        "PP-StructureV3": Model.PP_STRUCTURE_V3,
        "PaddleOCR-VL": Model.PADDLE_OCR_VL,
        "PaddleOCR-VL-1.5": Model.PADDLE_OCR_VL_15,
        "PaddleOCR-VL-1.6": Model.PADDLE_OCR_VL_16,
    }

    # Pipeline type classification
    # OCR pipelines return text recognition results
    _OCR_PIPELINES = {"OCR"}
    # Document parsing pipelines return structured markdown and image mappings
    _DOC_PARSING_PIPELINES = {
        "PP-StructureV3",
        "PaddleOCR-VL",
        "PaddleOCR-VL-1.5",
        "PaddleOCR-VL-1.6",
    }

    def __init__(
        self,
        pipeline: str,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 300.0,
        poll_timeout: float = 600.0,
    ):
        self._pipeline = pipeline
        self._token = token
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
        self._client: Optional[AsyncPaddleOCRClient] = None

        if pipeline not in self._PIPELINE_MODEL_MAP:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    async def start(self) -> None:
        try:
            self._client = AsyncPaddleOCRClient(
                token=self._token,
                base_url=self._base_url,
                request_timeout=self._request_timeout,
                poll_timeout=self._poll_timeout,
            )
        except AuthError as e:
            raise AuthenticationError(f"Authentication failed: {e}")

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _resolve_model(self) -> Model:
        """Get Model enum value for the pipeline"""
        return self._PIPELINE_MODEL_MAP[self._pipeline]

    def _resolve_input_source(self, input_data: str) -> Dict[str, str]:
        """Resolve input source (URL or file path).

        Args:
            input_data: Either a URL (http:// or https://) or a local file path.

        Returns:
            Dict with either 'file_url' key for URLs or 'file_path' key for local paths.
        """
        if input_data.startswith("http://") or input_data.startswith("https://"):
            return {"file_url": input_data}
        else:
            return {"file_path": input_data}

    async def execute(
        self, input_data: str, file_type: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        # Note: file_type is unused for AIStudio executor (parameter kept for base class consistency)
        if not self._client:
            raise RuntimeError("Executor not started")

        model = self._resolve_model()
        input_source = self._resolve_input_source(input_data)

        try:
            if self._pipeline in self._OCR_PIPELINES:
                # OCR call
                ocr_options = OCROptions(
                    use_doc_orientation_classify=options.get(
                        "use_doc_orientation_classify"
                    ),
                    use_doc_unwarping=options.get("use_doc_unwarping"),
                    visualize=False,  # Always False as MCP tool does not return visualization
                )
                result = await self._client.ocr(
                    model=model,
                    **input_source,
                    options=ocr_options,
                )
                return self._parse_ocr_result(result)

            elif self._pipeline in self._DOC_PARSING_PIPELINES:
                # Document parsing call
                if self._pipeline == "PP-StructureV3":
                    doc_options = PPStructureV3Options(
                        use_doc_unwarping=options.get("use_doc_unwarping"),
                        use_doc_orientation_classify=options.get(
                            "use_doc_orientation_classify"
                        ),
                        use_chart_recognition=options.get("use_chart_recognition"),
                        prettify_markdown=options.get("prettify_markdown"),
                    )
                else:  # PaddleOCR-VL series
                    doc_options = PaddleOCRVLOptions(
                        use_doc_unwarping=options.get("use_doc_unwarping"),
                        use_doc_orientation_classify=options.get(
                            "use_doc_orientation_classify"
                        ),
                        use_layout_detection=options.get("use_layout_detection"),
                        use_seal_recognition=options.get("use_seal_recognition"),
                        use_chart_recognition=options.get("use_chart_recognition"),
                        prettify_markdown=options.get("prettify_markdown"),
                    )
                result = await self._client.parse_document(
                    model=model,
                    **input_source,
                    options=doc_options,
                )
                return self._parse_doc_parsing_result(result)

        except AuthError as e:
            raise AuthenticationError(f"Authentication failed: {e}")
        except ServiceUnavailableError as e:
            raise ResourceUnavailableError(f"Service unavailable: {e}")
        except (JobFailedError, APIError, ResponseFormatError, ResultParseError) as e:
            raise ExecutorError(f"Execution failed: {e}")
        except RequestTimeoutError as e:
            raise ExecutionTimeoutError(f"Request timeout: {e}")

    def _parse_ocr_result(self, result) -> Dict[str, Any]:
        """Parse SDK OCRResult into unified format.

        Args:
            result: OCRResult object from AsyncPaddleOCRClient containing page results.

        Returns:
            Dict with keys:
                - text: Concatenated text from all detected lines
                - confidence: Average confidence across all lines
                - text_lines: List of dicts with 'text', 'confidence', and 'bbox' for each line
        """
        clean_texts, confidences, text_lines = [], [], []

        for page_result in result.pages:
            for line in page_result.text_lines:
                text = line.text
                conf = line.confidence
                bbox = line.bounding_box

                if text and text.strip():
                    clean_texts.append(text.strip())
                    confidences.append(conf)
                    text_lines.append(
                        {
                            "text": text.strip(),
                            "confidence": round(conf, 3),
                            "bbox": bbox,
                        }
                    )

        return {
            "text": "\n".join(clean_texts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "text_lines": text_lines,
        }

    def _parse_doc_parsing_result(self, result) -> Dict[str, Any]:
        """Parse SDK DocParsingResult into unified format.

        Args:
            result: DocParsingResult object from AsyncPaddleOCRClient containing page results.

        Returns:
            Dict with keys:
                - markdown: Concatenated markdown from all pages
                - pages: Total number of pages processed
                - images_mapping: Dict mapping image keys to their URLs across all pages
        """
        markdown_parts = []
        all_images_mapping = {}

        for page in result.pages:
            markdown_parts.append(page.markdown_text)
            # Process images from markdown_images dict
            for img_key, img_url in page.markdown_images.items():
                all_images_mapping[img_key] = img_url

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(result.pages),
            "images_mapping": all_images_mapping,
        }
