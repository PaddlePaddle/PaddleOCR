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
)

DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


def _raise_for_response(response: requests.Response) -> None:
    if response.status_code == 200:
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
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"Connection failed: {e}")
        _raise_for_response(resp)
        return resp.json()["data"]["jobId"]

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
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"Connection failed: {e}")
        _raise_for_response(resp)
        return resp.json()["data"]["jobId"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            resp = self._session.get(
                f"{self._base_url}/{job_id}",
                timeout=self._timeout,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"Connection failed: {e}")
        _raise_for_response(resp)
        return resp.json()["data"]

    def fetch_jsonl(self, url: str) -> list:
        # Result URLs are often pre-signed object storage links; do not send API token.
        try:
            resp = requests.get(url, timeout=self._timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"Connection failed: {e}")
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        results = []
        for line in lines:
            line = line.strip()
            if line:
                results.append(json.loads(line))
        return results

    def close(self):
        self._session.close()
