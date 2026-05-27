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
import contextlib
import json
import os
from typing import Optional, Union

import aiohttp

from ._core import (
    default_payload,
    extract_api_message_from_payload,
    extract_job_id,
    job_id_for_task,
    job_status_from_data,
    parse_batch_status,
    raise_for_status,
    resolve_document_model,
    resolve_document_options,
    resolve_ocr_model,
    unwrap_api_response,
    validate_input_source,
    validate_result_json_url,
    validate_state,
)
from .errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    JobFailedError,
    NetworkError,
    PollTimeoutError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
)
from .models import (
    DocParsingOptions,
    Model,
    OCROptions,
)
from ._resources import (
    save_document_parsing_result_resources,
    save_ocr_result_resources,
    save_resource,
)
from .results import BatchStatus, DocParsingResult, Job, JobStatus, OCRResult
from ._http import API_PATH, DEFAULT_BASE_URL
from ._poller import (
    DEFAULT_INITIAL_INTERVAL,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_MULTIPLIER,
    parse_doc_parsing_result,
    parse_ocr_result,
)


class AsyncPaddleOCRClient:
    """Async client for PaddleOCR API using aiohttp.

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
        self._base_url = resolved_base_url.rstrip("/")
        self._jobs_url = f"{self._base_url}{API_PATH}"
        if timeout is not None:
            request_timeout = timeout
            poll_timeout = timeout
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
        self._client_platform = client_platform
        self._session = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=self._api_headers(),
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _api_headers(self) -> dict:
        headers = {"Authorization": f"Bearer {self._token}"}
        if self._client_platform:
            headers["Client-Platform"] = self._client_platform
        return headers

    @contextlib.asynccontextmanager
    async def _request_scope(self):
        """Wraps an async block with unified error classification."""
        try:
            yield
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthError,
                    InvalidRequestError,
                    APIError,
                    FileNotFoundError,
                    RequestTimeoutError,
                    ResponseFormatError,
                    ResultParseError,
                    PollTimeoutError,
                    JobFailedError,
                    NetworkError,
                ),
            ):
                raise
            raise NetworkError(f"Connection failed: {e}") from e

    async def ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV5,
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
        jsonl_data = await self._poll_until_done(job_id)
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
        jsonl_data = await self._poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    async def submit_ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV5,
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

    async def get_status(self, job_id: str) -> JobStatus:
        await self._ensure_session()
        data = await self._get_job_status(job_id)
        return job_status_from_data(job_id, data)

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        await self._ensure_session()
        data = await self._get_batch_status(batch_id)
        return parse_batch_status(batch_id, data)

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
            timeout=self._request_timeout,
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
            timeout=self._request_timeout,
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
            timeout=self._request_timeout,
        )

    async def wait_ocr_result(self, job: Union[Job, str]) -> OCRResult:
        job_id = job_id_for_task(job, "ocr")
        jsonl_data = await self._poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    async def wait_document_parsing_result(
        self, job: Union[Job, str]
    ) -> DocParsingResult:
        job_id = job_id_for_task(job, "document_parsing")
        jsonl_data = await self._poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

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
        await self._ensure_session()
        payload = options.to_payload() if options else default_payload(model)
        if file_url:
            return await self._submit_url(
                model.value,
                file_url,
                payload,
                page_ranges=page_ranges,
                batch_id=batch_id,
            )
        return await self._submit_file(
            model.value,
            file_path,
            payload,
            page_ranges=page_ranges,
            batch_id=batch_id,
        )

    async def _submit_url(
        self,
        model: str,
        file_url: str,
        payload: dict,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> str:
        body = {
            "fileUrl": file_url,
            "model": model,
            "optionalPayload": payload,
        }
        if page_ranges is not None:
            body["pageRanges"] = page_ranges
        if batch_id is not None:
            body["batchId"] = batch_id
        async with self._request_scope():
            async with self._session.post(
                self._jobs_url,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                await self._raise_for_response(resp)
                data = await self._response_data(resp)
                return extract_job_id(data)

    async def _submit_file(
        self,
        model: str,
        file_path: str,
        payload: dict,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("optionalPayload", json.dumps(payload))
        if page_ranges is not None:
            form.add_field("pageRanges", page_ranges)
        if batch_id is not None:
            form.add_field("batchId", batch_id)
        with open(file_path, "rb") as f:
            file_data = f.read()
        form.add_field(
            "file",
            file_data,
            filename=os.path.basename(file_path),
        )
        async with self._request_scope():
            async with self._session.post(self._jobs_url, data=form) as resp:
                await self._raise_for_response(resp)
                data = await self._response_data(resp)
                return extract_job_id(data)

    async def _get_job_status(self, job_id: str) -> dict:
        async with self._request_scope():
            async with self._session.get(f"{self._jobs_url}/{job_id}") as resp:
                await self._raise_for_response(resp)
                return await self._response_data(resp)

    async def _get_batch_status(self, batch_id: str) -> dict:
        async with self._request_scope():
            async with self._session.get(f"{self._jobs_url}/batch/{batch_id}") as resp:
                await self._raise_for_response(resp)
                return await self._response_data(resp)

    async def _fetch_jsonl(self, url: str) -> list:
        async with self._request_scope():
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as bare_session:
                async with bare_session.get(url) as resp:
                    await self._raise_for_response(resp)
                    text = await resp.text()
                    try:
                        lines = text.strip().split("\n")
                        return [json.loads(line) for line in lines if line.strip()]
                    except json.JSONDecodeError as e:
                        raise ResultParseError(
                            f"Malformed JSONL result payload: {e}"
                        ) from e

    async def _poll_until_done(self, job_id: str) -> list:
        interval = DEFAULT_INITIAL_INTERVAL
        loop = asyncio.get_running_loop()
        start = loop.time()
        deadline = start + self._poll_timeout

        while True:
            now = loop.time()
            if now >= deadline:
                raise PollTimeoutError(job_id, now - start)

            data = await self._get_job_status(job_id)
            state = validate_state(data)

            if state == "done":
                json_url = validate_result_json_url(data)
                return await self._fetch_jsonl(json_url)

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise PollTimeoutError(job_id, loop.time() - start)
            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * DEFAULT_MULTIPLIER, DEFAULT_MAX_INTERVAL)

    async def _raise_for_response(self, resp) -> None:
        if 200 <= resp.status < 300:
            return
        try:
            body = await resp.json()
            msg = (
                extract_api_message_from_payload(body)
                if isinstance(body, dict)
                else None
            )
            if not msg:
                msg = await resp.text()
        except Exception:
            msg = await resp.text()
        raise_for_status(resp.status, msg)

    async def _response_data(self, resp) -> dict:
        try:
            payload = await resp.json()
        except Exception as e:
            raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
        return unwrap_api_response(payload, resp.status)
