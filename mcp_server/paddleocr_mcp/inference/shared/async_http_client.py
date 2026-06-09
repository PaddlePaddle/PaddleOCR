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

from typing import Any, Optional

import httpx


class AsyncHTTPClient:
    def __init__(
        self,
        base_url: str,
        http_timeout: int = 600,
        headers: Optional[dict[str, str]] = None,
    ):
        self._base_url = base_url
        self._http_timeout = http_timeout
        self._headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        write_timeout = min(float(self._http_timeout), 120.0)
        timeout = httpx.Timeout(
            connect=30.0,
            read=float(self._http_timeout),
            write=write_timeout,
            pool=30.0,
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("HTTP client not started")

        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        merged_headers = {**self._headers, **(headers or {})}

        response = await self._client.post(url, json=payload, headers=merged_headers)
        response.raise_for_status()
        return response.json()
