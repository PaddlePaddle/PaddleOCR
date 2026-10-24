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

from ._core import (
    job_status_from_data,
    parse_batch_status,
    validate_result_json_url,
    validate_state,
)
from .errors import (
    JobFailedError,
    PollTimeoutError,
    ResultParseError,
)
from .results import (
    BatchStatus,
    DocParsingPage,
    DocParsingResult,
    JobStatus,
    OCRPage,
    OCRResult,
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
                raise PollTimeoutError(job_id, now - start)

            data = self._http.get_job_status(job_id)
            state = validate_state(data)

            if state == "done":
                json_url = validate_result_json_url(data)
                jsonl_data = self._http.fetch_jsonl(json_url)
                return jsonl_data, data

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PollTimeoutError(job_id, time.monotonic() - start)

            time.sleep(min(interval, remaining))
            interval = min(interval * self._multiplier, self._max_interval)

    def get_status(self, job_id: str) -> JobStatus:
        data = self._http.get_job_status(job_id)
        return job_status_from_data(job_id, data)

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        data = self._http.get_batch_status(batch_id)
        return parse_batch_status(batch_id, data)


def parse_ocr_result(job_id: str, jsonl_data: list) -> OCRResult:
    try:
        pages = []
        data_info = {}
        for line_obj in jsonl_data:
            result = line_obj["result"]
            if isinstance(result.get("dataInfo"), dict):
                data_info.update(result["dataInfo"])
            for item in result["ocrResults"]:
                pages.append(
                    OCRPage(
                        pruned_result=item["prunedResult"],
                        ocr_image_url=item.get("ocrImage"),
                        doc_preprocessing_image_url=item.get("docPreprocessingImage"),
                        input_image_url=item.get("inputImage"),
                        raw=item,
                    )
                )
        return OCRResult(
            job_id=job_id,
            pages=pages,
            data_info=data_info,
        )
    except (KeyError, TypeError) as e:
        raise ResultParseError(f"Malformed OCR result payload: {e}") from e


def parse_doc_parsing_result(job_id: str, jsonl_data: list) -> DocParsingResult:
    try:
        pages = []
        data_info = {}
        for line_obj in jsonl_data:
            result = line_obj["result"]
            if isinstance(result.get("dataInfo"), dict):
                data_info.update(result["dataInfo"])
            for item in result["layoutParsingResults"]:
                markdown = item["markdown"]
                pages.append(
                    DocParsingPage(
                        markdown_text=markdown["text"],
                        markdown_images=markdown.get("images", {}),
                        output_images=item.get("outputImages", {}),
                        pruned_result=item.get("prunedResult"),
                        input_image_url=item.get("inputImage"),
                        exports=item.get("exports", {}),
                        markdown=markdown,
                        raw=item,
                    )
                )
        return DocParsingResult(
            job_id=job_id,
            pages=pages,
            data_info=data_info,
        )
    except (KeyError, TypeError) as e:
        raise ResultParseError(f"Malformed document parsing result payload: {e}") from e
