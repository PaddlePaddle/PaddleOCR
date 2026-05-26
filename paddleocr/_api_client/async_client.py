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
import json
import os
from typing import Any, Dict, List, Optional, Union

from ._core import (
    default_payload,
    job_id_for_task,
    resolve_document_model,
    resolve_document_options,
    resolve_ocr_model,
    validate_input_source,
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
from .results import BatchStatus, DocParsingResult, Job, JobStatus, OCRResult, Progress
from ._http import DEFAULT_BASE_URL
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
        base_url: str = DEFAULT_BASE_URL,
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
        self._base_url = base_url.rstrip("/")
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
            try:
                import aiohttp
            except ImportError:
                raise ImportError(
                    "aiohttp is required for AsyncPaddleOCRClient. "
                    "Install it with: pip install aiohttp>=3.8.0"
                )
            self._session = aiohttp.ClientSession(
                headers=self._api_headers(),
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _api_headers(self) -> dict:
        headers = {"Authorization": f"bearer {self._token}"}
        if self._client_platform:
            headers["Client-Platform"] = self._client_platform
        return headers

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
        return self._job_status_from_data(job_id, data)

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        await self._ensure_session()
        data = await self._get_batch_status(batch_id)
        result = data.get("extractResult")
        if not isinstance(result, list):
            raise ResponseFormatError(
                "Batch response data must contain list 'extractResult'."
            )
        jobs = []
        for item in result:
            if not isinstance(item, dict):
                raise ResponseFormatError("Batch extractResult items must be objects.")
            job_id = item.get("jobId")
            if not isinstance(job_id, str) or not job_id:
                raise ResponseFormatError(
                    "Batch extractResult items must contain non-empty string 'jobId'."
                )
            jobs.append(self._job_status_from_data(job_id, item))
        return BatchStatus(batch_id=batch_id, jobs=jobs)

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
        try:
            async with self._session.post(
                self._base_url,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                await self._raise_for_response(resp)
                data = await self._response_data(resp)
                return self._job_id_from_data(data)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthError,
                    InvalidRequestError,
                    APIError,
                    RequestTimeoutError,
                    ResponseFormatError,
                ),
            ):
                raise
            raise NetworkError(f"Connection failed: {e}") from e

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
        try:
            import aiohttp

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
            async with self._session.post(self._base_url, data=form) as resp:
                await self._raise_for_response(resp)
                data = await self._response_data(resp)
                return self._job_id_from_data(data)
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
                ),
            ):
                raise
            raise NetworkError(f"Connection failed: {e}") from e

    async def _get_job_status(self, job_id: str) -> dict:
        try:
            async with self._session.get(f"{self._base_url}/{job_id}") as resp:
                await self._raise_for_response(resp)
                return await self._response_data(resp)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthError,
                    InvalidRequestError,
                    APIError,
                    RequestTimeoutError,
                    ResponseFormatError,
                ),
            ):
                raise
            raise NetworkError(f"Connection failed: {e}") from e

    async def _get_batch_status(self, batch_id: str) -> dict:
        try:
            async with self._session.get(f"{self._base_url}/batch/{batch_id}") as resp:
                await self._raise_for_response(resp)
                return await self._response_data(resp)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthError,
                    InvalidRequestError,
                    APIError,
                    RequestTimeoutError,
                    ResponseFormatError,
                ),
            ):
                raise
            raise NetworkError(f"Connection failed: {e}") from e

    async def _fetch_jsonl(self, url: str) -> list:
        # Result URLs are often pre-signed object storage links.
        try:
            import aiohttp

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
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            if isinstance(e, (AuthError, InvalidRequestError, APIError)):
                raise
            if isinstance(e, (RequestTimeoutError, ResultParseError)):
                raise
            raise NetworkError(f"Failed to fetch result: {e}") from e

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
            state = self._validate_state(data)

            if state == "done":
                result_url = data.get("resultUrl")
                if not isinstance(result_url, dict):
                    raise ResponseFormatError(
                        "Done job response must contain object 'resultUrl'."
                    )
                json_url = result_url.get("jsonUrl")
                if not isinstance(json_url, str) or not json_url:
                    raise ResponseFormatError(
                        "Done job response resultUrl must contain non-empty string 'jsonUrl'."
                    )
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
            msg = self._extract_api_message(body) or await resp.text()
        except Exception:
            msg = await resp.text()
        if resp.status in (401, 403):
            raise AuthError(f"Authentication failed: {msg}")
        elif resp.status == 400:
            raise InvalidRequestError(f"Bad request: {msg}")
        else:
            raise APIError(resp.status, msg)

    def _job_status_from_data(self, job_id: str, data: dict) -> JobStatus:
        state = self._validate_state(data)
        progress = None
        ep = data.get("extractProgress")
        if ep:
            if not isinstance(ep, dict):
                raise ResponseFormatError("'extractProgress' must be an object.")
            progress = Progress(
                total_pages=ep.get("totalPages", 0),
                extracted_pages=ep.get("extractedPages", 0),
                start_time=ep.get("startTime"),
                end_time=ep.get("endTime"),
            )
        return JobStatus(
            job_id=job_id,
            state=state,
            progress=progress,
            result=data.get("resultUrl"),
            error_msg=data.get("errorMsg"),
        )

    def _validate_state(self, data: dict) -> str:
        state = data.get("state")
        if state not in {"pending", "running", "done", "failed"}:
            raise ResponseFormatError(f"Unknown or missing job state: {state}")
        return state

    async def _response_data(self, resp) -> dict:
        try:
            payload = await resp.json()
        except Exception as e:
            raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise ResponseFormatError("Response body must be a JSON object.")
        code = payload.get("code", 0)
        if code not in (0, None):
            raise APIError(resp.status, self._extract_api_message(payload) or "")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ResponseFormatError("Response JSON must contain object field 'data'.")
        return data

    def _job_id_from_data(self, data: dict) -> str:
        job_id = data.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise ResponseFormatError(
                "Response data must contain non-empty string 'jobId'."
            )
        return job_id

    def _extract_api_message(self, payload: dict) -> Optional[str]:
        for key in ("msg", "errorMsg", "message"):
            value = payload.get(key)
            if value:
                return str(value)
        data = payload.get("data")
        if isinstance(data, dict):
            value = data.get("errorMsg")
            if value:
                return str(value)
        return None
