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
import time
from typing import Optional, Union

from ._http import (
    DEFAULT_BASE_URL,
    _response_data,
    _validate_job_id,
    _validate_result_url,
    _validate_state,
)
from ._poller import (
    DEFAULT_INITIAL_INTERVAL,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_MULTIPLIER,
    parse_doc_parsing_result,
    parse_ocr_result,
)
from .errors import (
    APIError,
    AuthError,
    FileNotFoundError,
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
    is_document_parsing_model,
    is_ocr_model,
)
from .results import DocParsingResult, Job, JobStatus, OCRResult, Progress


class AsyncAPIClient:
    """Async client for PaddleOCR official API using aiohttp.

    Supports asyncio.gather for concurrent job submission and polling.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout: float = 300.0,
        poll_timeout: float = 600.0,
    ):
        self._token = token or os.environ.get("PADDLEOCR_ACCESS_TOKEN", "")
        if not self._token:
            raise AuthError(
                "Token is required. Set PADDLEOCR_ACCESS_TOKEN or pass token=."
            )
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
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
                    "aiohttp is required for AsyncAPIClient. "
                    "Install it with: pip install aiohttp>=3.8.0"
                )
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"bearer {self._token}"},
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV5,
    ) -> OCRResult:
        model = self._ocr_model(model)
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
        model: Union[Model, str] = Model.PADDLE_OCR_VL_15,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> DocParsingResult:
        model = self._document_model(model)
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
        model = self._ocr_model(model)
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
        model: Union[Model, str] = Model.PADDLE_OCR_VL_15,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Job:
        model = self._document_model(model)
        job_id = await self._submit(
            model, file_url, file_path, options, page_ranges, batch_id
        )
        return Job(job_id=job_id, model=model.value, task="document_parsing")

    async def get_status(self, job_id: str) -> JobStatus:
        await self._ensure_session()
        data = await self._get_job_status(job_id)
        state = _validate_state(data)
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
            error_msg=data.get("errorMsg"),
        )

    async def wait_ocr_result(self, job: Union[Job, str]) -> OCRResult:
        job_id = self._job_id_for_task(job, "ocr")
        jsonl_data = await self._poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    async def wait_document_parsing_result(
        self, job: Union[Job, str]
    ) -> DocParsingResult:
        job_id = self._job_id_for_task(job, "document_parsing")
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
        if not file_url and not file_path:
            raise InvalidRequestError("Either file_url or file_path is required.")
        if file_url and file_path:
            raise InvalidRequestError("file_url and file_path are mutually exclusive.")
        await self._ensure_session()
        payload = options.to_payload() if options else self._default_payload(model)
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
                data = _response_data(await self._response_json(resp))
                return _validate_job_id(data)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except _aiohttp_client_error() as e:
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
                # aiohttp streams file objects from FormData; keep the handle
                # open until the request context has completed.
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                )
                async with self._session.post(self._base_url, data=form) as resp:
                    await self._raise_for_response(resp)
                    data = _response_data(await self._response_json(resp))
                    return _validate_job_id(data)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except _aiohttp_client_error() as e:
            raise NetworkError(f"Connection failed: {e}") from e

    async def _get_job_status(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> dict:
        try:
            request_kwargs = {}
            if timeout is not None:
                import aiohttp

                request_kwargs["timeout"] = aiohttp.ClientTimeout(
                    total=min(self._request_timeout, timeout)
                )
            async with self._session.get(
                f"{self._base_url}/{job_id}",
                **request_kwargs,
            ) as resp:
                await self._raise_for_response(resp)
                data = _response_data(await self._response_json(resp))
                _validate_state(data)
                return data
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except _aiohttp_client_error() as e:
            raise NetworkError(f"Connection failed: {e}") from e

    async def _fetch_jsonl(self, url: str) -> list:
        # Result URLs are often pre-signed object storage links; do not send API token.
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as bare_session:
                async with bare_session.get(url) as resp:
                    await self._raise_for_response(resp)
                    text = await resp.text()
                    return self._parse_jsonl_text(text)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except _aiohttp_client_error() as e:
            raise NetworkError(f"Failed to fetch result: {e}") from e

    async def _poll_until_done(self, job_id: str) -> list:
        interval = DEFAULT_INITIAL_INTERVAL
        start = time.monotonic()
        deadline = start + self._poll_timeout

        while True:
            now = time.monotonic()
            if now >= deadline:
                raise PollTimeoutError(job_id, now - start)

            data = await self._get_job_status(job_id, timeout=deadline - now)
            now = time.monotonic()
            if now >= deadline:
                raise PollTimeoutError(job_id, now - start)

            state = _validate_state(data)

            if state == "done":
                json_url = _validate_result_url(data)
                return await self._fetch_jsonl(json_url)

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            remaining = deadline - now
            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * DEFAULT_MULTIPLIER, DEFAULT_MAX_INTERVAL)

    async def _raise_for_response(self, resp) -> None:
        if 200 <= resp.status < 300:
            return
        try:
            body = await resp.json()
            msg = body.get("message", await resp.text())
        except Exception:
            msg = await resp.text()
        if resp.status in (401, 403):
            raise AuthError(f"Authentication failed: {msg}")
        elif resp.status == 400:
            raise InvalidRequestError(f"Bad request: {msg}")
        else:
            raise APIError(resp.status, msg)

    async def _response_json(self, resp) -> dict:
        try:
            payload = await resp.json()
        except _aiohttp_content_type_error() as e:
            raise ResponseFormatError(
                f"Response body is not valid JSON: {type(e).__name__}"
            ) from e
        except ValueError as e:
            raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise ResponseFormatError("Response body must be a JSON object.")
        return payload

    def _parse_jsonl_text(self, text: str) -> list:
        lines = text.strip().split("\n")
        results = []
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ResultParseError(
                        f"Malformed JSONL at line {line_number}: {e}"
                    ) from e
        return results

    def _default_payload(self, model: Model) -> dict:
        if is_ocr_model(model):
            return OCROptions().to_payload()
        return DocParsingOptions().to_payload()

    def _job_id_for_task(self, job: Union[Job, str], expected_task: str) -> str:
        if isinstance(job, str):
            return job
        if job.task != expected_task:
            raise InvalidRequestError(
                f"Job {job.job_id} is for task {job.task}, not {expected_task}."
            )
        job_is_ocr_model = is_ocr_model(job.model)
        if expected_task == "ocr" and not job_is_ocr_model:
            raise InvalidRequestError(
                f"Job {job.job_id} model {job.model} is not valid for OCR."
            )
        if expected_task == "document_parsing" and not is_document_parsing_model(
            job.model
        ):
            raise InvalidRequestError(
                f"Job {job.job_id} model {job.model} is not valid for document parsing."
            )
        return job.job_id

    def _ocr_model(self, model: Union[Model, str]) -> Model:
        try:
            resolved = model if isinstance(model, Model) else Model(model)
        except ValueError as e:
            raise InvalidRequestError(f"Unsupported model: {model}") from e
        if not is_ocr_model(resolved):
            raise InvalidRequestError(
                f"{resolved.value} is not an OCR model and cannot be used for OCR."
            )
        return resolved

    def _document_model(self, model: Union[Model, str]) -> Model:
        try:
            resolved = model if isinstance(model, Model) else Model(model)
        except ValueError as e:
            raise InvalidRequestError(f"Unsupported model: {model}") from e
        if is_ocr_model(resolved):
            raise InvalidRequestError(
                f"{resolved.value} is an OCR model and cannot be used for document parsing."
            )
        if not is_document_parsing_model(resolved):
            raise InvalidRequestError(
                f"Unsupported document parsing model: {resolved.value}"
            )
        return resolved


def _aiohttp_client_error():
    try:
        import aiohttp
    except ImportError:
        return ()
    return (aiohttp.ClientError,)


def _aiohttp_content_type_error():
    try:
        import aiohttp
    except ImportError:
        return ()
    return (aiohttp.ContentTypeError,)
