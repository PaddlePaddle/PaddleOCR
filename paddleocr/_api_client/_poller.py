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

import time
from typing import Any

from .errors import (
    JobFailedError,
    PollTimeoutError,
    ResponseFormatError,
    ResultParseError,
)
from .results import (
    BatchStatus,
    DocParsingPage,
    DocParsingResult,
    JobStatus,
    OCRPage,
    OCRResult,
    Progress,
)

DEFAULT_INITIAL_INTERVAL = 3.0
DEFAULT_MULTIPLIER = 1.5
DEFAULT_MAX_INTERVAL = 15.0
DEFAULT_MAX_WAIT_TIME = 600.0


class Poller:
    def __init__(
        self,
        http_client,
        initial_interval: float = DEFAULT_INITIAL_INTERVAL,
        multiplier: float = DEFAULT_MULTIPLIER,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        max_wait_time: float = DEFAULT_MAX_WAIT_TIME,
    ):
        self._http = http_client
        self._initial_interval = initial_interval
        self._multiplier = multiplier
        self._max_interval = max_interval
        self._max_wait_time = max_wait_time

    def poll_until_done(self, job_id: str) -> Any:
        interval = self._initial_interval
        start = time.monotonic()
        deadline = start + self._max_wait_time

        while True:
            now = time.monotonic()
            if now >= deadline:
                raise PollTimeoutError(
                    f"Timed out after {now - start:.1f}s waiting for job {job_id}"
                )

            data = self._http.get_job_status(job_id)
            state = _validate_state(data)

            if state == "done":
                json_url = _validate_json_url(data)
                jsonl_data = self._http.fetch_jsonl(json_url)
                return jsonl_data, data

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise PollTimeoutError(
                    f"Timed out after {now - start:.1f}s waiting for job {job_id}"
                )
            time.sleep(min(interval, remaining))
            interval = min(interval * self._multiplier, self._max_interval)

    def get_status(self, job_id: str) -> JobStatus:
        data = self._http.get_job_status(job_id)
        return _job_status_from_data(job_id, data)

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        data = self._http.get_batch_status(batch_id)
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
            jobs.append(_job_status_from_data(job_id, item))
        return BatchStatus(batch_id=batch_id, jobs=jobs)


def _job_status_from_data(job_id: str, data: dict) -> JobStatus:
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
        result=data.get("resultUrl"),
        error_msg=data.get("errorMsg"),
    )


def _validate_state(data: dict) -> str:
    state = data.get("state")
    if state not in {"pending", "running", "done", "failed"}:
        raise ResponseFormatError(f"Unknown or missing job state: {state}")
    return state


def _validate_json_url(data: dict) -> str:
    result_url = data.get("resultUrl")
    if not isinstance(result_url, dict):
        raise ResponseFormatError("Done job response must contain object 'resultUrl'.")
    json_url = result_url.get("jsonUrl")
    if not isinstance(json_url, str) or not json_url:
        raise ResponseFormatError(
            "Done job response resultUrl must contain non-empty string 'jsonUrl'."
        )
    return json_url


def parse_ocr_result(job_id: str, jsonl_data: list) -> OCRResult:
    try:
        pages = []
        for line_obj in jsonl_data:
            result = line_obj["result"]
            for item in result["ocrResults"]:
                pages.append(
                    OCRPage(
                        pruned_result=item["prunedResult"],
                        ocr_image_url=item.get("ocrImage"),
                    )
                )
        return OCRResult(job_id=job_id, pages=pages)
    except (KeyError, TypeError) as e:
        raise ResultParseError(f"Malformed OCR result payload: {e}") from e


def parse_doc_parsing_result(job_id: str, jsonl_data: list) -> DocParsingResult:
    try:
        pages = []
        for line_obj in jsonl_data:
            result = line_obj["result"]
            for item in result["layoutParsingResults"]:
                markdown = item["markdown"]
                pages.append(
                    DocParsingPage(
                        markdown_text=markdown["text"],
                        markdown_images=markdown.get("images", {}),
                        output_images=item.get("outputImages", {}),
                    )
                )
        return DocParsingResult(job_id=job_id, pages=pages)
    except (KeyError, TypeError) as e:
        raise ResultParseError(f"Malformed document parsing result payload: {e}") from e
