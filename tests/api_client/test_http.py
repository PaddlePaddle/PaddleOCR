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
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from paddleocr._api_client._http import API_PATH, DEFAULT_BASE_URL, HTTPClient
from paddleocr._api_client.client import PaddleOCRClient
from paddleocr._api_client.errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    ServiceUnavailableError,
)
from paddleocr._api_client.models import Model


class _MockHandler(BaseHTTPRequestHandler):
    """Records requests and responds with canned JSON."""

    requests_log = []
    response_body = {"code": 0, "data": {"jobId": "job-1"}}
    response_status = 200

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        self.__class__.requests_log.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )
        self.send_response(self.__class__.response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.__class__.response_body).encode())

    def do_GET(self):
        self.__class__.requests_log.append(
            {"method": "GET", "path": self.path, "headers": dict(self.headers)}
        )
        self.send_response(self.__class__.response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.__class__.response_body).encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def mock_server():
    _MockHandler.requests_log = []
    _MockHandler.response_body = {"code": 0, "data": {"jobId": "job-1"}}
    _MockHandler.response_status = 200

    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", _MockHandler
    server.shutdown()


def test_base_url_appends_api_path(mock_server):
    base_url, handler = mock_server
    client = HTTPClient("test-token", base_url, timeout=5.0)

    client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})

    assert len(handler.requests_log) == 1
    assert handler.requests_log[0]["path"] == API_PATH


def test_get_job_status_url(mock_server):
    base_url, handler = mock_server
    handler.response_body = {"code": 0, "data": {"jobId": "job-1", "state": "running"}}
    client = HTTPClient("test-token", base_url, timeout=5.0)

    client.get_job_status("job-123")

    assert handler.requests_log[0]["path"] == f"{API_PATH}/job-123"


def test_get_batch_status_url(mock_server):
    base_url, handler = mock_server
    handler.response_body = {"code": 0, "data": {"batchId": "b-1", "jobs": []}}
    client = HTTPClient("test-token", base_url, timeout=5.0)

    client.get_batch_status("batch-abc")

    assert handler.requests_log[0]["path"] == f"{API_PATH}/batch/batch-abc"


def test_base_url_trailing_slashes_normalized(mock_server):
    base_url, handler = mock_server
    client = HTTPClient("test-token", base_url + "///", timeout=5.0)

    client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})

    assert handler.requests_log[0]["path"] == API_PATH


def test_authorization_header_sent(mock_server):
    base_url, handler = mock_server
    client = HTTPClient("my-secret-token", base_url, timeout=5.0)

    client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})

    assert (
        handler.requests_log[0]["headers"]["Authorization"] == "Bearer my-secret-token"
    )


def test_client_platform_header(mock_server):
    base_url, handler = mock_server
    client = HTTPClient("token", base_url, timeout=5.0, client_platform="sdk-test")

    client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})

    assert handler.requests_log[0]["headers"]["Client-Platform"] == "sdk-test"


def test_submit_url_posts_correct_body(mock_server):
    base_url, handler = mock_server
    client = HTTPClient("token", base_url, timeout=5.0)

    job_id = client.submit_url(
        "PP-OCRv5",
        "https://example.test/file.pdf",
        {"visualize": True},
        page_ranges="1-3",
        batch_id="batch-1",
    )

    assert job_id == "job-1"
    body = json.loads(handler.requests_log[0]["body"])
    assert body == {
        "fileUrl": "https://example.test/file.pdf",
        "model": "PP-OCRv5",
        "optionalPayload": {"visualize": True},
        "pageRanges": "1-3",
        "batchId": "batch-1",
    }


def test_error_mapping_401(mock_server):
    base_url, handler = mock_server
    handler.response_status = 401
    handler.response_body = {"code": 401, "msg": "Unauthorized"}
    client = HTTPClient("bad-token", base_url, timeout=5.0)

    with pytest.raises(AuthError):
        client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})


def test_error_mapping_400(mock_server):
    base_url, handler = mock_server
    handler.response_status = 400
    handler.response_body = {"code": 400, "msg": "Bad request"}
    client = HTTPClient("token", base_url, timeout=5.0)

    with pytest.raises(InvalidRequestError):
        client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})


def test_error_mapping_429(mock_server):
    base_url, handler = mock_server
    handler.response_status = 429
    handler.response_body = {"code": 429, "msg": "Rate limited"}
    client = HTTPClient("token", base_url, timeout=5.0)

    with pytest.raises(RateLimitError):
        client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})


def test_error_mapping_503(mock_server):
    base_url, handler = mock_server
    handler.response_status = 503
    handler.response_body = {"code": 503, "msg": "Service unavailable"}
    client = HTTPClient("token", base_url, timeout=5.0)

    with pytest.raises(ServiceUnavailableError):
        client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})


def test_error_mapping_500(mock_server):
    base_url, handler = mock_server
    handler.response_status = 500
    handler.response_body = {"code": 500, "msg": "Internal error"}
    client = HTTPClient("token", base_url, timeout=5.0)

    with pytest.raises(APIError):
        client.submit_url("PP-OCRv5", "https://example.test/file.pdf", {})


def test_paddleocr_client_requires_token(monkeypatch):
    monkeypatch.delenv("PADDLEOCR_ACCESS_TOKEN", raising=False)
    with pytest.raises(AuthError):
        PaddleOCRClient()


def test_paddleocr_client_reads_env_token(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "env-token")
    client = PaddleOCRClient()
    client.close()


def test_default_base_url_is_host_only():
    assert "/api/" not in DEFAULT_BASE_URL
    assert DEFAULT_BASE_URL == "https://paddleocr.aistudio-app.com"


def test_paddleocr_client_client_platform_header(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"code": 0, "data": {"jobId": "job-1"}}

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.captured_headers = None

        def post(self, *args, **kwargs):
            self.captured_headers = dict(self.headers)
            return FakeResponse()

        def close(self):
            return None

    fake_session = FakeSession()
    monkeypatch.setattr(
        "paddleocr._api_client._http.requests.Session",
        lambda: fake_session,
    )

    client = PaddleOCRClient(token="token", client_platform="my-app")
    job = client.submit_ocr(file_url="https://example.test/input.pdf")

    assert job.job_id == "job-1"
    assert fake_session.captured_headers["Client-Platform"] == "my-app"
