import argparse
from pathlib import Path

import pytest

from paddleocr._api_client import cli
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


def test_client_platform_sets_header_on_api_requests(monkeypatch):
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


def test_cli_save_resources_uses_result_specific_helper(tmp_path, monkeypatch, capsys):
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, **kwargs):
            return OCRResult(
                job_id="job-1",
                pages=[
                    OCRPage(
                        pruned_result={}, ocr_image_url="https://example.test/a.png"
                    )
                ],
            )

        def save_ocr_result_resources(self, result, destination, overwrite=False):
            assert result.job_id == "job-1"
            assert destination == str(tmp_path)
            assert overwrite is True
            return [str(Path(destination) / "ocr-page-1.png")]

        def close(self):
            return None

    monkeypatch.setattr(cli, "PaddleOCRClient", FakeClient)

    args = argparse.Namespace(
        token="token",
        client_platform=None,
        base_url=None,
        request_timeout=300.0,
        poll_timeout=600.0,
        model=None,
        model_type="ocr",
        file_url="https://example.test/input.pdf",
        file_path=None,
        output=None,
        save_resources=str(tmp_path),
        overwrite_resources=True,
        page_ranges=None,
        batch_id=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        use_chart_recognition=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        use_formula_recognition=None,
        use_layout_detection=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_rec_score_thresh=None,
        prettify_markdown=None,
        visualize=None,
    )

    cli._execute_api(args)

    captured = capsys.readouterr()
    assert '"jobId": "job-1"' in captured.out
    assert "Resources saved to:" in captured.err
