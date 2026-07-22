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

"""Async HTTP client transport-error mapping.

The async client must honor the same contract as the sync client: every network
failure surfaces as a ``PaddleOCRAPIError`` subclass, never a raw
``aiohttp``/``asyncio`` exception. These tests use ``asyncio.run`` so they need
no extra async-plugin dependency.
"""

import asyncio
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from paddleocr._api_client._async_http import AsyncHTTPClient
from paddleocr._api_client.errors import NetworkError, RequestTimeoutError


class _SlowHandler(BaseHTTPRequestHandler):
    """Sleeps before responding, to force a client-side timeout."""

    delay = 2.0

    def do_GET(self):
        time.sleep(self.__class__.delay)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"code": 0, "data": {}}')

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def slow_server():
    server = HTTPServer(("127.0.0.1", 0), _SlowHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _unused_url():
    # Bind then immediately release a port so a connection there is refused.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def test_connection_error_maps_to_network_error():
    client = AsyncHTTPClient(token="t", base_url=_unused_url(), timeout=5.0)

    async def run():
        try:
            with pytest.raises(NetworkError):
                await client.get_job_status("job-1")
        finally:
            await client.close()

    asyncio.run(run())


def test_timeout_maps_to_request_timeout_error(slow_server):
    client = AsyncHTTPClient(token="t", base_url=slow_server, timeout=0.5)

    async def run():
        try:
            with pytest.raises(RequestTimeoutError):
                await client.get_job_status("job-1")
        finally:
            await client.close()

    asyncio.run(run())


def test_fetch_jsonl_connection_error_maps_to_network_error():
    client = AsyncHTTPClient(token="t", base_url="http://127.0.0.1", timeout=5.0)

    async def run():
        try:
            with pytest.raises(NetworkError):
                await client.fetch_jsonl(_unused_url())
        finally:
            await client.close()

    asyncio.run(run())
