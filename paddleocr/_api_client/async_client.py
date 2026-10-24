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

import asyncio
import os
from typing import Optional, Union

from ._core import (
    default_payload,
    job_id_for_task,
    resolve_document_model,
    resolve_document_options,
    resolve_ocr_model,
    validate_input_source,
)
from ._async_http import AsyncHTTPClient
from ._http import DEFAULT_BASE_URL
from ._async_poller import AsyncPoller
from ._poller import parse_doc_parsing_result, parse_ocr_result
from ._resources import (
    save_document_parsing_result_resources,
    save_ocr_result_resources,
    save_resource,
)
from .errors import AuthError
from .models import (
    DocParsingOptions,
    Model,
    OCROptions,
)
from .results import BatchStatus, DocParsingResult, Job, JobStatus, OCRResult


class AsyncPaddleOCRClient:
    """Async client for PaddleOCR API.

    Supports asyncio.gather for concurrent job submission and polling.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 300.0,
        poll_timeout: float = 600.0,
        timeout: Optional[float] = None,
        client_platform: Optional[str] = None,
    ):
        self._token = token or os.environ.get("PADDLEOCR_ACCESS_TOKEN", "")
        if not self._token:
            raise AuthError(
                "Token is required. Set PADDLEOCR_ACCESS_TOKEN or pass token=."
            )
        resolved_base_url = (
            base_url or os.environ.get("PADDLEOCR_BASE_URL") or DEFAULT_BASE_URL
        )
        if timeout is not None:
            request_timeout = timeout
            poll_timeout = timeout
        self._http = AsyncHTTPClient(
            self._token,
            resolved_base_url,
            request_timeout,
            client_platform=client_platform,
        )
        self._poller = AsyncPoller(self._http, max_wait_time=poll_timeout)

    async def __aenter__(self):
        await self._http.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._http.close()

    async def close(self):
        await self._http.close()

    async def ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV6,
    ) -> OCRResult:
        model = resolve_ocr_model(model)
        job_id = await self._submit(
            model,
            file_url,
            file_path,
            options,
            page_ranges,
            batch_id,
        )
        jsonl_data, _ = await self._poller.poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    async def parse_document(
        self,
        model: Union[Model, str] = Model.PADDLE_OCR_VL_16,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> DocParsingResult:
        model = resolve_document_model(model)
        options = resolve_document_options(model, options)
        job_id = await self._submit(
            model, file_url, file_path, options, page_ranges, batch_id
        )
        jsonl_data, _ = await self._poller.poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    async def submit_ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV6,
    ) -> Job:
        model = resolve_ocr_model(model)
        job_id = await self._submit(
            model,
            file_url,
            file_path,
            options,
            page_ranges,
            batch_id,
        )
        return Job(job_id=job_id, model=model.value, task="ocr")

    async def submit_document_parsing(
        self,
        model: Union[Model, str] = Model.PADDLE_OCR_VL_16,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Job:
        model = resolve_document_model(model)
        options = resolve_document_options(model, options)
        job_id = await self._submit(
            model, file_url, file_path, options, page_ranges, batch_id
        )
        return Job(job_id=job_id, model=model.value, task="document_parsing")

    async def wait_ocr_result(self, job: Union[Job, str]) -> OCRResult:
        job_id = job_id_for_task(job, "ocr")
        jsonl_data, _ = await self._poller.poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    async def wait_document_parsing_result(
        self, job: Union[Job, str]
    ) -> DocParsingResult:
        job_id = job_id_for_task(job, "document_parsing")
        jsonl_data, _ = await self._poller.poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    async def get_status(self, job_id: str) -> JobStatus:
        return await self._poller.get_status(job_id)

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        return await self._poller.get_batch_status(batch_id)

    async def save_resource(
        self,
        resource_url: str,
        destination: str,
        *,
        overwrite: bool = False,
        filename: Optional[str] = None,
    ) -> str:
        return await asyncio.to_thread(
            save_resource,
            resource_url,
            destination,
            overwrite=overwrite,
            filename=filename,
            timeout=self._http.timeout,
        )

    async def save_ocr_result_resources(
        self,
        result: OCRResult,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> list:
        return await asyncio.to_thread(
            save_ocr_result_resources,
            result,
            destination,
            overwrite=overwrite,
            timeout=self._http.timeout,
        )

    async def save_document_parsing_result_resources(
        self,
        result: DocParsingResult,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> list:
        return await asyncio.to_thread(
            save_document_parsing_result_resources,
            result,
            destination,
            overwrite=overwrite,
            timeout=self._http.timeout,
        )

    async def _submit(
        self,
        model: Model,
        file_url: Optional[str],
        file_path: Optional[str],
        options,
        page_ranges: Optional[str],
        batch_id: Optional[str],
    ) -> str:
        validate_input_source(file_url, file_path)
        payload = options.to_payload() if options else default_payload(model)
        if file_url:
            return await self._http.submit_url(
                model.value,
                file_url,
                payload,
                page_ranges=page_ranges,
                batch_id=batch_id,
            )
        return await self._http.submit_file(
            model.value,
            file_path,
            payload,
            page_ranges=page_ranges,
            batch_id=batch_id,
        )
