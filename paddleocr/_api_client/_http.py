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

from .errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    NetworkError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
)

DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


def _raise_for_response(response: requests.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    msg = _extract_api_message(response)
    if response.status_code in (401, 403):
        raise AuthError(f"Authentication failed: {msg}")
    if response.status_code == 400:
        raise InvalidRequestError(f"Bad request: {msg}")
    raise APIError(response.status_code, msg)


def _extract_api_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        for key in ("msg", "errorMsg", "message"):
            value = payload.get(key)
            if value:
                return str(value)
        data = payload.get("data")
        if isinstance(data, dict):
            value = data.get("errorMsg")
            if value:
                return str(value)
    return response.text


def _response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as e:
        raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ResponseFormatError("Response body must be a JSON object.")
    code = payload.get("code", 0)
    if code not in (0, None):
        raise APIError(response.status_code, _extract_api_message(response))
    return payload


def _response_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ResponseFormatError("Response JSON must contain object field 'data'.")
    return data


def _job_id_from_response(response: requests.Response) -> str:
    data = _response_data(_response_json(response))
    job_id = data.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise ResponseFormatError(
            "Response data must contain non-empty string 'jobId'."
        )
    return job_id


class HTTPClient:
    def __init__(self, token: str, base_url: str, timeout: float):
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"bearer {token}"

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
                self._base_url,
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
                    self._base_url,
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
                f"{self._base_url}/{job_id}",
                timeout=self._timeout,
            )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        return _response_data(_response_json(resp))

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        try:
            resp = self._session.get(
                f"{self._base_url}/batch/{batch_id}",
                timeout=self._timeout,
            )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        return _response_data(_response_json(resp))

    def fetch_jsonl(self, url: str) -> list:
        # Result URLs are often pre-signed object storage links; do not send API token.
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
