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

import requests

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

DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com"
API_PATH = "/api/v2/ocr/jobs"


def _raise_for_response(response: requests.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    raise_for_status(response.status_code, _extract_api_message(response))


def _extract_api_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        msg = extract_api_message_from_payload(payload)
        if msg:
            return msg
    return response.text


def _response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as e:
        raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ResponseFormatError("Response body must be a JSON object.")
    return payload


def _response_data(response: requests.Response) -> Dict[str, Any]:
    payload = _response_json(response)
    return unwrap_api_response(payload, response.status_code)


def _job_id_from_response(response: requests.Response) -> str:
    return extract_job_id(_response_data(response))


class HTTPClient:
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
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"
        if client_platform:
            self._session.headers["Client-Platform"] = client_platform

    @property
    def timeout(self) -> float:
        return self._timeout

    def submit_url(
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
        try:
            resp = self._session.post(
                self._jobs_url,
                json=body,
                timeout=self._timeout,
            )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        return _job_id_from_response(resp)

    def submit_file(
        self,
        model: str,
        file_path: str,
        optional_payload: dict,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        data = {
            "model": model,
            "optionalPayload": json.dumps(optional_payload),
        }
        if page_ranges is not None:
            data["pageRanges"] = page_ranges
        if batch_id is not None:
            data["batchId"] = batch_id
        try:
            with open(file_path, "rb") as f:
                resp = self._session.post(
                    self._jobs_url,
                    data=data,
                    files={"file": f},
                    timeout=self._timeout,
                )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        return _job_id_from_response(resp)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            resp = self._session.get(
                f"{self._jobs_url}/{job_id}",
                timeout=self._timeout,
            )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        return _response_data(resp)

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        try:
            resp = self._session.get(
                f"{self._jobs_url}/batch/{batch_id}",
                timeout=self._timeout,
            )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        return _response_data(resp)

    def fetch_jsonl(self, url: str) -> list:
        # Result URLs are often pre-signed object storage links.
        try:
            resp = requests.get(url, timeout=self._timeout)
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        try:
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            results = []
            for line in lines:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results
        except json.JSONDecodeError as e:
            raise ResultParseError(f"Malformed JSONL result payload: {e}") from e

    def close(self):
        self._session.close()
