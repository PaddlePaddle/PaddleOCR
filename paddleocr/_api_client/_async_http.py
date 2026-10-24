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

import json
import os
from typing import Any, Dict, Optional

import aiohttp

from ._core import (
    extract_api_message_from_payload,
    extract_job_id,
    raise_for_status,
    unwrap_api_response,
)
from .errors import (
    NetworkError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
)
from ._http import API_PATH, DEFAULT_BASE_URL


class AsyncHTTPClient:
    def __init__(
        self,
        token: str,
        base_url: str,
        timeout: float,
        client_platform: Optional[str] = None,
    ):
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._jobs_url = f"{self._base_url}{API_PATH}"
        self._timeout = timeout
        self._client_platform = client_platform
        self._session = None

    @property
    def timeout(self) -> float:
        return self._timeout

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=self._api_headers(),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
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

    async def submit_url(
        self,
        model: str,
        file_url: str,
        optional_payload: dict,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> str:
        body = {
            "fileUrl": file_url,
            "model": model,
            "optionalPayload": optional_payload,
        }
        if page_ranges is not None:
            body["pageRanges"] = page_ranges
        if batch_id is not None:
            body["batchId"] = batch_id

        await self._ensure_session()
        async with self._session.post(
            self._jobs_url,
            json=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            await self._raise_for_response(resp)
            data = await self._response_data(resp)
            return extract_job_id(data)

    async def submit_file(
        self,
        model: str,
        file_path: str,
        optional_payload: dict,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("optionalPayload", json.dumps(optional_payload))
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

        await self._ensure_session()
        async with self._session.post(self._jobs_url, data=form) as resp:
            await self._raise_for_response(resp)
            data = await self._response_data(resp)
            return extract_job_id(data)

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        await self._ensure_session()
        async with self._session.get(f"{self._jobs_url}/{job_id}") as resp:
            await self._raise_for_response(resp)
            return await self._response_data(resp)

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        await self._ensure_session()
        async with self._session.get(f"{self._jobs_url}/batch/{batch_id}") as resp:
            await self._raise_for_response(resp)
            return await self._response_data(resp)

    async def fetch_jsonl(self, url: str) -> list:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
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

    async def _response_data(self, resp) -> Dict[str, Any]:
        try:
            payload = await resp.json()
        except Exception as e:
            raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
        return unwrap_api_response(payload, resp.status)
