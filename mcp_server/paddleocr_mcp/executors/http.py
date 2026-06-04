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

import abc
import asyncio
import json
from typing import Any, Dict, Optional

import httpx

from .base import Executor, AuthenticationError, ExecutorError, ResourceUnavailableError


class HTTPExecutor(Executor):
    """Abstract base class for synchronous HTTP APIs, encapsulating common HTTP call logic"""

    def __init__(self, base_url: str, timeout: int = 60):
        self._base_url = base_url
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        timeout = httpx.Timeout(connect=30.0, read=self._timeout, write=30.0, pool=30.0)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post(
        self, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Execute HTTP POST request"""
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            response = await self._client.post(url, json=payload, headers=headers)
            if response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {response.text}")
            if response.status_code == 503:
                raise ResourceUnavailableError(f"Service unavailable: {response.text}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ExecutorError(f"HTTP request failed: {e}")
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise ExecutorError(f"HTTP request failed: {e}")

    @abc.abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers (subclass implements authentication)"""

    @abc.abstractmethod
    def _get_endpoint(self) -> str:
        """Get API endpoint"""

    @abc.abstractmethod
    def _prepare_payload(
        self, input_data: str, file_type: Optional[str], **options
    ) -> Dict[str, Any]:
        """Prepare request payload (subclass implements)"""

    @abc.abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response into unified format (subclass implements)"""

    def _parse_ocr_response_http(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OCR response from HTTP API (shared implementation)"""
        ocr_results = result_data.get("ocrResults", [])
        all_texts, all_confidences, text_lines = [], [], []

        for ocr_result in ocr_results:
            pruned = ocr_result["prunedResult"]
            texts = pruned["rec_texts"]
            scores = pruned["rec_scores"]
            boxes = pruned["rec_boxes"]

            for i, text in enumerate(texts):
                if text and text.strip():
                    conf = scores[i] if i < len(scores) else 0
                    all_texts.append(text.strip())
                    all_confidences.append(conf)
                    text_lines.append(
                        {
                            "text": text.strip(),
                            "confidence": round(conf, 3),
                            "bbox": boxes[i],
                        }
                    )

        return {
            "text": "\n".join(all_texts),
            "confidence": (
                sum(all_confidences) / len(all_confidences) if all_confidences else 0
            ),
            "text_lines": text_lines,
        }

    def _parse_doc_parsing_response_http(
        self, result_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse document parsing response from HTTP API (shared implementation)"""
        doc_parsing_results = result_data.get("layoutParsingResults", [])
        markdown_parts = []
        all_images_mapping = {}

        for res in doc_parsing_results:
            markdown_parts.append(res["markdown"]["text"])
            images = res["markdown"]["images"]
            all_images_mapping.update(images)

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(doc_parsing_results),
            "images_mapping": all_images_mapping,
        }

    async def execute(
        self, input_data: str, file_type: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        headers = self._get_headers()
        payload = self._prepare_payload(input_data, file_type, **options)
        response = await self._post(self._get_endpoint(), payload, headers)
        return self._parse_response(response)
