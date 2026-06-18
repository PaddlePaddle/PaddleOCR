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

from pathlib import Path

import pytest

from paddleocr._api_client.client import PaddleOCRClient
from paddleocr._api_client.errors import InvalidRequestError
from paddleocr._api_client.results import (
    DocParsingPage,
    DocParsingResult,
    OCRPage,
    OCRResult,
)


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_save_resource_writes_single_url(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _Response(b"content")

    monkeypatch.setattr("paddleocr._api_client._resources.requests.get", fake_get)

    client = PaddleOCRClient(token="token", request_timeout=12.0)
    saved_path = client.save_resource(
        "https://example.test/path/image.png", str(tmp_path)
    )

    assert saved_path == str(tmp_path / "image.png")
    assert (tmp_path / "image.png").read_bytes() == b"content"
    assert calls == [("https://example.test/path/image.png", 12.0)]


def test_save_ocr_result_resources_uses_stable_page_names(tmp_path, monkeypatch):
    contents = [b"one", b"two"]

    def fake_get(url, timeout):
        return _Response(contents.pop(0))

    monkeypatch.setattr("paddleocr._api_client._resources.requests.get", fake_get)

    client = PaddleOCRClient(token="token")
    result = OCRResult(
        job_id="job-1",
        pages=[
            OCRPage(pruned_result={}, ocr_image_url="https://example.test/a.png?sig=1"),
            OCRPage(pruned_result={}, ocr_image_url="https://example.test/b.jpg"),
        ],
    )

    saved_paths = client.save_ocr_result_resources(result, str(tmp_path))

    assert saved_paths == [
        str(tmp_path / "ocr-page-1.png"),
        str(tmp_path / "ocr-page-2.jpg"),
    ]
    assert (tmp_path / "ocr-page-1.png").read_bytes() == b"one"
    assert (tmp_path / "ocr-page-2.jpg").read_bytes() == b"two"


def test_save_document_parsing_result_resources_sorts_and_validates_keys(
    tmp_path, monkeypatch
):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _Response(url.encode())

    monkeypatch.setattr("paddleocr._api_client._resources.requests.get", fake_get)

    client = PaddleOCRClient(token="token")
    result = DocParsingResult(
        job_id="job-1",
        pages=[
            DocParsingPage(
                markdown_text="",
                markdown_images={
                    "b.png": "https://example.test/b",
                    "a.png": "https://example.test/a",
                },
                output_images={"rendered.jpg": "https://example.test/rendered"},
            )
        ],
    )

    saved_paths = client.save_document_parsing_result_resources(result, str(tmp_path))

    assert saved_paths == [
        str(tmp_path / "a.png"),
        str(tmp_path / "b.png"),
        str(tmp_path / "rendered.jpg"),
    ]
    assert calls == [
        "https://example.test/a",
        "https://example.test/b",
        "https://example.test/rendered",
    ]

    for unsafe_key in ("../escape.png", "nested\\escape.png"):
        bad_result = DocParsingResult(
            job_id="job-2",
            pages=[
                DocParsingPage(
                    markdown_text="",
                    markdown_images={unsafe_key: "https://example.test/escape"},
                )
            ],
        )
        with pytest.raises(InvalidRequestError):
            client.save_document_parsing_result_resources(bad_result, str(tmp_path))
