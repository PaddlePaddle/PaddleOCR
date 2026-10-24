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
)
from .results import BatchStatus, JobStatus

DEFAULT_INITIAL_INTERVAL = 3.0
DEFAULT_MULTIPLIER = 1.5
DEFAULT_MAX_INTERVAL = 15.0
DEFAULT_MAX_WAIT_TIME = 600.0


class AsyncPoller:
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

    async def poll_until_done(self, job_id: str) -> Any:
        interval = self._initial_interval
        loop = asyncio.get_running_loop()
        start = loop.time()
        deadline = start + self._max_wait_time

        while True:
            now = loop.time()
            if now >= deadline:
                raise PollTimeoutError(job_id, now - start)

            data = await self._http.get_job_status(job_id)
            state = validate_state(data)

            if state == "done":
                json_url = validate_result_json_url(data)
                jsonl_data = await self._http.fetch_jsonl(json_url)
                return jsonl_data, data

            if state == "failed":
                error_msg = data.get("errorMsg", "Unknown error")
                raise JobFailedError(job_id, error_msg)

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise PollTimeoutError(job_id, loop.time() - start)
            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * self._multiplier, self._max_interval)

    async def get_status(self, job_id: str) -> JobStatus:
        data = await self._http.get_job_status(job_id)
        return job_status_from_data(job_id, data)

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        data = await self._http.get_batch_status(batch_id)
        return parse_batch_status(batch_id, data)
