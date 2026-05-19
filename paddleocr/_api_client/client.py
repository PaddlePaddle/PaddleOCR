# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import Optional, Union

from ._http import DEFAULT_BASE_URL, HTTPClient
from ._poller import Poller, parse_doc_parsing_result, parse_ocr_result
from .errors import AuthError, InvalidRequestError
from .models import DocParsingOptions, Model, OCROptions
from .results import DocParsingResult, Job, JobStatus, OCRResult


class APIClient:
    """Synchronous blocking client for PaddleOCR API.

    Wraps the async job API internally: submit → poll → fetch result.
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
        self._http = HTTPClient(self._token, base_url, timeout)
        self._poller = Poller(self._http, max_wait_time=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._http.close()

    def ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
    ) -> OCRResult:
        job_id = self._submit(Model.PP_OCRV5, file_url, file_path, options)
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    def doc_parsing(
        self,
        model: Model,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
    ) -> DocParsingResult:
        job_id = self._submit(model, file_url, file_path, options)
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    def submit_ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
    ) -> Job:
        job_id = self._submit(Model.PP_OCRV5, file_url, file_path, options)
        return Job(job_id=job_id)

    def submit_doc_parsing(
        self,
        model: Model,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
    ) -> Job:
        job_id = self._submit(model, file_url, file_path, options)
        return Job(job_id=job_id)

    def wait_for_result(
        self, job_id: str
    ) -> Union[OCRResult, DocParsingResult]:
        jsonl_data, data = self._poller.poll_until_done(job_id)
        return self._parse_result(job_id, jsonl_data)

    def get_result(self, job_id: str) -> JobStatus:
        return self._poller.get_status(job_id)

    def _submit(
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
        payload = options.to_payload() if options else self._default_payload(model)
        if file_url:
            return self._http.submit_url(model.value, file_url, payload)
        return self._http.submit_file(model.value, file_path, payload)

    def _default_payload(self, model: Model) -> dict:
        if model == Model.PP_OCRV5:
            return OCROptions().to_payload()
        return DocParsingOptions().to_payload()

    def _parse_result(self, job_id: str, jsonl_data: list):
        first = jsonl_data[0]["result"] if jsonl_data else {}
        if "ocrResults" in first:
            return parse_ocr_result(job_id, jsonl_data)
        return parse_doc_parsing_result(job_id, jsonl_data)
