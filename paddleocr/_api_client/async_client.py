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

from .errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    JobFailedError,
    NetworkError,
    TimeoutError,
)
from .models import DocParsingOptions, Model, OCROptions
from .results import DocParsingResult, Job, JobStatus, OCRResult, Progress
from ._poller import (
    DEFAULT_INITIAL_INTERVAL,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_MAX_WAIT_TIME,
    DEFAULT_MULTIPLIER,
    parse_doc_parsing_result,
    parse_ocr_result,
)

DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


class AsyncAPIClient:
    """Async client for PaddleOCR API using aiohttp.

    Supports asyncio.gather for concurrent job submission and polling.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
    ):
        self._token = token or os.environ.get("PADDLE_OCR_TOKEN", "")
        if not self._token:
            raise AuthError("Token is required. Set PADDLE_OCR_TOKEN or pass token=.")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
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
                timeout=aiohttp.ClientTimeout(total=self._timeout),
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
    ) -> OCRResult:
        job_id = await self._submit(Model.PP_OCRV5, file_url, file_path, options)
        jsonl_data = await self._poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    async def doc_parsing(
        self,
        model: Model,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
    ) -> DocParsingResult:
        job_id = await self._submit(model, file_url, file_path, options)
        jsonl_data = await self._poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    async def submit_ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
    ) -> Job:
        job_id = await self._submit(Model.PP_OCRV5, file_url, file_path, options)
        return Job(job_id=job_id)

    async def submit_doc_parsing(
        self,
        model: Model,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
    ) -> Job:
        job_id = await self._submit(model, file_url, file_path, options)
        return Job(job_id=job_id)

    async def wait_for_result(
        self, job_id: str
    ) -> Union[OCRResult, DocParsingResult]:
        jsonl_data = await self._poll_until_done(job_id)
        first = jsonl_data[0]["result"] if jsonl_data else {}
        if "ocrResults" in first:
            return parse_ocr_result(job_id, jsonl_data)
        return parse_doc_parsing_result(job_id, jsonl_data)

    async def get_result(self, job_id: str) -> JobStatus:
        await self._ensure_session()
        data = await self._get_job_status(job_id)
        progress = None
        ep = data.get("extractProgress")
        if ep:
            progress = Progress(
                total_pages=ep.get("totalPages", 0),
                extracted_pages=ep.get("extractedPages", 0),
                start_time=ep.get("startTime"),
                end_time=ep.get("endTime"),
            )
        return JobStatus(
            job_id=job_id,
            state=data["state"],
            progress=progress,
            error_msg=data.get("errorMsg"),
        )

    async def _submit(
        self,
        model: Model,
        file_url: Optional[str],
        file_path: Optional[str],
        options,
    ) -> str:
        if not file_url and not file_path:
            raise InvalidRequestError("Either file_url or file_path is required.")
        if file_url and file_path:
            raise InvalidRequestError(
                "file_url and file_path are mutually exclusive."
            )
        await self._ensure_session()
        payload = options.to_payload() if options else self._default_payload(model)
        if file_url:
            return await self._submit_url(model.value, file_url, payload)
        return await self._submit_file(model.value, file_path, payload)

    async def _submit_url(self, model: str, file_url: str, payload: dict) -> str:
        body = {
            "fileUrl": file_url,
            "model": model,
            "optionalPayload": payload,
        }
        try:
            async with self._session.post(
                self._base_url,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                await self._raise_for_response(resp)
                data = await resp.json()
                return data["data"]["jobId"]
        except Exception as e:
            if isinstance(e, (AuthError, InvalidRequestError, APIError)):
                raise
            raise NetworkError(f"Connection failed: {e}")

    async def _submit_file(self, model: str, file_path: str, payload: dict) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        try:
            import aiohttp

            form = aiohttp.FormData()
            form.add_field("model", model)
            form.add_field("optionalPayload", json.dumps(payload))
            with open(file_path, "rb") as f:
                file_data = f.read()
            form.add_field(
                "file",
                file_data,
                filename=os.path.basename(file_path),
            )
            async with self._session.post(self._base_url, data=form) as resp:
                await self._raise_for_response(resp)
                data = await resp.json()
                return data["data"]["jobId"]
        except Exception as e:
            if isinstance(e, (AuthError, InvalidRequestError, APIError, FileNotFoundError)):
                raise
            raise NetworkError(f"Connection failed: {e}")

    async def _get_job_status(self, job_id: str) -> dict:
        try:
            async with self._session.get(
                f"{self._base_url}/{job_id}"
            ) as resp:
                await self._raise_for_response(resp)
                data = await resp.json()
                return data["data"]
        except Exception as e:
            if isinstance(e, (AuthError, APIError)):
                raise
            raise NetworkError(f"Connection failed: {e}")

    async def _fetch_jsonl(self, url: str) -> list:
        try:
            async with self._session.get(url) as resp:
                text = await resp.text()
                lines = text.strip().split("\n")
                return [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            raise NetworkError(f"Failed to fetch result: {e}")

    async def _poll_until_done(self, job_id: str) -> list:
        interval = DEFAULT_INITIAL_INTERVAL
        elapsed = 0.0

        while elapsed < DEFAULT_MAX_WAIT_TIME:
            await asyncio.sleep(interval)
            elapsed += interval

            data = await self._get_job_status(job_id)
            state = data["state"]

            if state == "done":
                json_url = data["resultUrl"]["jsonUrl"]
                return await self._fetch_jsonl(json_url)

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            interval = min(interval * DEFAULT_MULTIPLIER, DEFAULT_MAX_INTERVAL)

        raise TimeoutError(job_id, elapsed)

    async def _raise_for_response(self, resp) -> None:
        if resp.status == 200:
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

    def _default_payload(self, model: Model) -> dict:
        if model == Model.PP_OCRV5:
            return OCROptions().to_payload()
        return DocParsingOptions().to_payload()
