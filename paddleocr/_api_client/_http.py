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
    FileNotFoundError,
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
    try:
        msg = response.json().get("message", response.text)
    except Exception:
        msg = response.text
    if response.status_code in (401, 403):
        raise AuthError(f"Authentication failed: {msg}")
    elif response.status_code == 400:
        raise InvalidRequestError(f"Bad request: {msg}")
    else:
        raise APIError(response.status_code, msg)


def _response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as e:
        raise ResponseFormatError(f"Response body is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ResponseFormatError("Response body must be a JSON object.")
    return payload


def _response_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ResponseFormatError("Response JSON must contain object field 'data'.")
    return data


def _validate_job_id(data: Dict[str, Any]) -> str:
    job_id = data.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise ResponseFormatError(
            "Response data must contain non-empty string 'jobId'."
        )
    return job_id


_KNOWN_STATES = {"pending", "running", "done", "failed"}


def _validate_state(data: Dict[str, Any]) -> str:
    state = data.get("state")
    if not isinstance(state, str) or not state:
        raise ResponseFormatError(
            "Response data must contain non-empty string 'state'."
        )
    if state not in _KNOWN_STATES:
        raise ResponseFormatError(f"Unknown job state: {state}")
    return state


def _validate_result_url(data: Dict[str, Any]) -> str:
    result_url = data.get("resultUrl")
    if not isinstance(result_url, dict):
        raise ResponseFormatError("Done job response must contain object 'resultUrl'.")
    json_url = result_url.get("jsonUrl")
    if not isinstance(json_url, str) or not json_url:
        raise ResponseFormatError(
            "Done job response resultUrl must contain non-empty string 'jsonUrl'."
        )
    return json_url


def _extract_api_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("message", response.text))
    except ValueError:
        pass
    return response.text


def _job_id_from_response(response: requests.Response) -> str:
    return _validate_job_id(_response_data(_response_json(response)))


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

    def get_job_status(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        request_timeout = (
            self._timeout if timeout is None else min(self._timeout, timeout)
        )
        try:
            resp = self._session.get(
                f"{self._base_url}/{job_id}",
                timeout=request_timeout,
            )
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        _raise_for_response(resp)
        data = _response_data(_response_json(resp))
        _validate_state(data)
        return data

    def fetch_jsonl(self, url: str) -> list:
        # Result URLs are often pre-signed object storage links; do not send API token.
        try:
            resp = requests.get(url, timeout=self._timeout)
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        status_code = getattr(resp, "status_code", 200)
        if not isinstance(status_code, int):
            status_code = 200
        if not 200 <= status_code < 300:
            raise APIError(status_code, _extract_api_message(resp))
        lines = resp.text.strip().split("\n")
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

    def close(self):
        self._session.close()
