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
from typing import Any, Dict, List

from .errors import JobFailedError, TimeoutError
from .results import (
    DocParsingPage,
    DocParsingResult,
    Job,
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
        elapsed = 0.0

        while elapsed < self._max_wait_time:
            time.sleep(interval)
            elapsed += interval

            data = self._http.get_job_status(job_id)
            state = data["state"]

            if state == "done":
                json_url = data["resultUrl"]["jsonUrl"]
                jsonl_data = self._http.fetch_jsonl(json_url)
                return jsonl_data, data

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            interval = min(interval * self._multiplier, self._max_interval)

        raise TimeoutError(job_id, elapsed)

    def get_status(self, job_id: str) -> JobStatus:
        data = self._http.get_job_status(job_id)
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


def parse_ocr_result(job_id: str, jsonl_data: list) -> OCRResult:
    pages = []
    for line_obj in jsonl_data:
        result = line_obj["result"]
        for item in result["ocrResults"]:
            pages.append(
                OCRPage(
                    pruned_result=item["prunedResult"],
                    ocr_image_url=item["ocrImage"],
                )
            )
    return OCRResult(job_id=job_id, pages=pages)


def parse_doc_parsing_result(job_id: str, jsonl_data: list) -> DocParsingResult:
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
