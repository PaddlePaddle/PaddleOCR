import json
import os
import tempfile
import unittest
from argparse import Namespace
from typing import get_args
from unittest.mock import MagicMock, patch

import requests

from paddleocr import (
    APIClient,
    DocParsingOptions,
    DocParsingPage,
    DocParsingResult,
    Job,
    Model,
    OCROptions,
    OCRPage,
    OCRResult,
    Progress,
)
from paddleocr._api_client._http import HTTPClient, _raise_for_response
from paddleocr._api_client._poller import (
    Poller,
    parse_doc_parsing_result,
    parse_ocr_result,
)
from paddleocr._api_client.cli import _execute_api, register_api_command
from paddleocr._api_client.errors import (
    APIError,
    AuthError,
    FileNotFoundError,
    InvalidRequestError,
    JobFailedError,
    NetworkError,
    PaddleOCRAPIError,
    PollTimeoutError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
)
from paddleocr._api_client.models import is_document_parsing_model, is_ocr_model


class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self._json = json_data
        self.status_code = status_code
        self.text = text or json.dumps(json_data)

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class TestAPIClient(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLEOCR_ACCESS_TOKEN"] = "test-token"

    def tearDown(self):
        if "PADDLEOCR_ACCESS_TOKEN" in os.environ:
            del os.environ["PADDLEOCR_ACCESS_TOKEN"]

    def test_no_token_raises(self):
        if "PADDLEOCR_ACCESS_TOKEN" in os.environ:
            del os.environ["PADDLEOCR_ACCESS_TOKEN"]
        with self.assertRaises(AuthError):
            APIClient()

    def test_no_file_raises(self):
        client = APIClient()
        with self.assertRaises(InvalidRequestError):
            client.ocr()

    def test_both_file_raises(self):
        client = APIClient()
        with self.assertRaises(InvalidRequestError):
            client.ocr(file_url="http://x.com/f.pdf", file_path="./f.pdf")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_ocr_url_full_flow(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        # Submit response
        submit_resp = MockResponse({"data": {"jobId": "job-123"}})
        # Poll: pending -> done
        pending_resp = MockResponse(
            {
                "data": {
                    "state": "pending",
                    "extractProgress": {"totalPages": 1, "extractedPages": 0},
                }
            }
        )
        done_resp = MockResponse(
            {
                "data": {
                    "state": "done",
                    "extractProgress": {
                        "totalPages": 1,
                        "extractedPages": 1,
                        "startTime": "t1",
                        "endTime": "t2",
                    },
                    "resultUrl": {"jsonUrl": "http://result.url/data.jsonl"},
                }
            }
        )
        # JSONL fetch
        jsonl_text = json.dumps(
            {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {"text": "hello"},
                            "ocrImage": "http://img.url/1.jpg",
                        }
                    ]
                }
            }
        )
        jsonl_resp = MockResponse(None, text=jsonl_text)
        jsonl_resp.text = jsonl_text
        jsonl_resp.raise_for_status = lambda: None

        mock_session.post.return_value = submit_resp
        mock_session.get.side_effect = [pending_resp, done_resp]

        jsonl_resp.raise_for_status = lambda: None

        with patch("paddleocr._api_client._http.requests.get") as mock_req_get:
            mock_req_get.return_value = jsonl_resp
            client = APIClient()
            with patch("paddleocr._api_client._poller.time.sleep"):
                result = client.ocr(file_url="http://example.com/test.pdf")

            mock_req_get.assert_called_once()
            self.assertNotIn(
                "Authorization",
                mock_req_get.call_args.kwargs.get("headers") or {},
            )

        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].pruned_result, {"text": "hello"})
        self.assertEqual(result.pages[0].ocr_image_url, "http://img.url/1.jpg")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_parse_document_url(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        submit_resp = MockResponse({"data": {"jobId": "job-456"}})
        done_resp = MockResponse(
            {
                "data": {
                    "state": "done",
                    "extractProgress": {"totalPages": 1, "extractedPages": 1},
                    "resultUrl": {"jsonUrl": "http://result.url/data.jsonl"},
                }
            }
        )
        jsonl_text = json.dumps(
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "# Title",
                                "images": {"img1.png": "http://img/1.png"},
                            },
                            "outputImages": {"vis": "http://vis/1.jpg"},
                        }
                    ]
                }
            }
        )
        jsonl_resp = MockResponse(None, text=jsonl_text)
        jsonl_resp.text = jsonl_text
        jsonl_resp.raise_for_status = lambda: None

        mock_session.post.return_value = submit_resp
        mock_session.get.side_effect = [done_resp]

        jsonl_resp.raise_for_status = lambda: None

        with patch("paddleocr._api_client._http.requests.get") as mock_req_get:
            mock_req_get.return_value = jsonl_resp
            client = APIClient()
            with patch("paddleocr._api_client._poller.time.sleep"):
                result = client.parse_document(
                    model=Model.PP_STRUCTURE_V3,
                    file_url="http://example.com/doc.pdf",
                    options=DocParsingOptions(use_chart_recognition=True),
                )

            kwargs = mock_req_get.call_args.kwargs
            hdrs = kwargs.get("headers")
            self.assertFalse(hdrs and "Authorization" in hdrs)

        self.assertEqual(result.job_id, "job-456")
        self.assertEqual(result.pages[0].markdown_text, "# Title")
        self.assertEqual(
            result.pages[0].markdown_images, {"img1.png": "http://img/1.png"}
        )

    @patch("paddleocr._api_client._http.requests.Session")
    def test_job_failed(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        submit_resp = MockResponse({"data": {"jobId": "job-fail"}})
        failed_resp = MockResponse(
            {"data": {"state": "failed", "errorMsg": "File corrupted"}}
        )

        mock_session.post.return_value = submit_resp
        mock_session.get.return_value = failed_resp

        client = APIClient()
        with patch("paddleocr._api_client._poller.time.sleep"):
            with self.assertRaises(JobFailedError) as ctx:
                client.ocr(file_url="http://example.com/bad.pdf")
            self.assertIn("File corrupted", str(ctx.exception))

    def test_context_manager(self):
        with APIClient() as client:
            self.assertIsNotNone(client)

    def test_sync_client_exposes_only_contract_public_names(self):
        client = APIClient()
        self.assertFalse(hasattr(client, "get_result"))
        self.assertFalse(hasattr(client, "wait_for_result"))
        self.assertFalse(hasattr(client, "doc_parsing"))
        self.assertFalse(hasattr(client, "submit_doc_parsing"))
        self.assertTrue(hasattr(client, "parse_document"))
        self.assertTrue(hasattr(client, "submit_document_parsing"))
        self.assertFalse(hasattr(client, "wait_for_ocr_result"))
        self.assertTrue(hasattr(client, "wait_ocr_result"))

    def test_wait_ocr_result_accepts_job_and_parses_ocr_result(self):
        client = APIClient()
        client._poller.poll_until_done = MagicMock(
            return_value=(
                [
                    {
                        "result": {
                            "ocrResults": [
                                {
                                    "prunedResult": {"text": "hello"},
                                    "ocrImage": "http://img/ocr.png",
                                }
                            ]
                        }
                    }
                ],
                {"state": "done"},
            )
        )

        result = client.wait_ocr_result(
            Job(job_id="job-ocr", model=Model.PP_OCRV5.value, task="ocr")
        )

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.job_id, "job-ocr")
        self.assertEqual(result.pages[0].pruned_result, {"text": "hello"})

    def test_model_helpers_classify_current_ocr_model(self):
        self.assertTrue(is_ocr_model(Model.PP_OCRV5))
        self.assertTrue(is_ocr_model(Model.PP_OCRV5.value))
        self.assertFalse(is_ocr_model(Model.PP_STRUCTURE_V3))
        self.assertFalse(is_ocr_model("future-unknown-model"))
        self.assertTrue(is_document_parsing_model(Model.PP_STRUCTURE_V3))
        self.assertFalse(is_document_parsing_model(Model.PP_OCRV5))

    def test_wait_document_parsing_result_accepts_job_and_parses_doc_result(self):
        client = APIClient()
        client._poller.poll_until_done = MagicMock(
            return_value=(
                [
                    {
                        "result": {
                            "layoutParsingResults": [
                                {
                                    "markdown": {
                                        "text": "# Title",
                                        "images": {"img.png": "http://img/doc.png"},
                                    },
                                    "outputImages": {"page": "http://out/page.png"},
                                }
                            ]
                        }
                    }
                ],
                {"state": "done"},
            )
        )

        result = client.wait_document_parsing_result(
            Job(
                job_id="job-doc",
                model=Model.PP_STRUCTURE_V3.value,
                task="document_parsing",
            )
        )

        self.assertIsInstance(result, DocParsingResult)
        self.assertEqual(result.job_id, "job-doc")
        self.assertEqual(result.pages[0].markdown_text, "# Title")

    def test_wait_document_parsing_result_rejects_unknown_model(self):
        client = APIClient()
        client._poller.poll_until_done = MagicMock(
            return_value=(
                [
                    {
                        "result": {
                            "layoutParsingResults": [
                                {"markdown": {"text": "# Should not poll"}}
                            ]
                        }
                    }
                ],
                {"state": "done"},
            )
        )

        with self.assertRaises(InvalidRequestError):
            client.wait_document_parsing_result(
                Job(
                    job_id="job-doc",
                    model="future-unknown-model",
                    task="document_parsing",
                )
            )
        client._poller.poll_until_done.assert_not_called()

    def test_typed_wait_methods_reject_mismatched_jobs(self):
        client = APIClient()

        with self.assertRaises(InvalidRequestError):
            client.wait_ocr_result(
                Job(
                    job_id="job-doc",
                    model=Model.PP_STRUCTURE_V3.value,
                    task="document_parsing",
                )
            )

        with self.assertRaises(InvalidRequestError):
            client.wait_document_parsing_result(
                Job(job_id="job-ocr", model=Model.PP_OCRV5.value, task="ocr")
            )

    def test_typed_wait_accepts_bare_job_id_and_uses_method_parser(self):
        client = APIClient()
        client._poller.poll_until_done = MagicMock(
            return_value=(
                [
                    {
                        "result": {
                            "ocrResults": [
                                {
                                    "prunedResult": {"text": "from string"},
                                    "ocrImage": "http://img/string.png",
                                }
                            ]
                        }
                    }
                ],
                {"state": "done"},
            )
        )

        result = client.wait_ocr_result("job-string")

        self.assertEqual(result.job_id, "job-string")
        self.assertEqual(result.pages[0].pruned_result, {"text": "from string"})

    def test_models_payload(self):
        opts = OCROptions(
            use_doc_orientation_classify=True, use_textline_orientation=True
        )
        payload = opts.to_payload()
        self.assertEqual(
            payload,
            {
                "useDocOrientationClassify": True,
                "useTextlineOrientation": True,
            },
        )

        opts2 = DocParsingOptions(use_chart_recognition=True)
        payload2 = opts2.to_payload()
        self.assertEqual(
            payload2,
            {
                "useChartRecognition": True,
            },
        )

    @patch("paddleocr._api_client._http.requests.Session")
    def test_submit_includes_top_level_job_parameters(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse({"data": {"jobId": "job-789"}})

        client = APIClient()
        job = client.submit_ocr(
            file_url="http://example.com/test.pdf",
            page_ranges="1,3-4",
            batch_id="batch-1",
        )

        self.assertEqual(job.job_id, "job-789")
        self.assertEqual(job.model, Model.PP_OCRV5.value)
        self.assertEqual(job.task, "ocr")
        _, kwargs = mock_session.post.call_args
        self.assertEqual(kwargs["json"]["pageRanges"], "1,3-4")
        self.assertEqual(kwargs["json"]["batchId"], "batch-1")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_submit_ocr_propagates_explicit_model(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse({"data": {"jobId": "job-ocr"}})

        client = APIClient()
        job = client.submit_ocr(
            model=Model.PP_OCRV5,
            file_url="http://example.com/test.pdf",
        )

        self.assertEqual(job.model, Model.PP_OCRV5.value)
        self.assertEqual(
            mock_session.post.call_args.kwargs["json"]["model"], Model.PP_OCRV5.value
        )

    def test_submit_ocr_rejects_non_ocr_model(self):
        client = APIClient()
        client._submit = MagicMock()

        with self.assertRaises(InvalidRequestError):
            client.submit_ocr(
                model=Model.PP_STRUCTURE_V3,
                file_url="http://example.com/test.pdf",
            )

        client._submit.assert_not_called()

    @patch("paddleocr._api_client._http.requests.Session")
    def test_submit_document_parsing_job_carries_model_metadata(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse({"data": {"jobId": "job-doc"}})

        client = APIClient()
        job = client.submit_document_parsing(
            model=Model.PADDLE_OCR_VL,
            file_url="http://example.com/doc.pdf",
        )

        self.assertEqual(job.job_id, "job-doc")
        self.assertEqual(job.model, Model.PADDLE_OCR_VL.value)
        self.assertEqual(job.task, "document_parsing")

    def test_job_task_public_literal_uses_contract_names(self):
        task_values = set(get_args(Job.__annotations__["task"]))

        self.assertEqual(task_values, {"ocr", "document_parsing"})
        self.assertNotIn("doc_parsing", task_values)

    def test_document_parsing_rejects_ocr_model(self):
        client = APIClient()
        client._submit = MagicMock()

        with self.assertRaises(InvalidRequestError):
            client.submit_document_parsing(
                model=Model.PP_OCRV5,
                file_url="http://example.com/doc.pdf",
            )
        with self.assertRaises(InvalidRequestError):
            client.submit_document_parsing(
                model=Model.PP_OCRV5.value,
                file_url="http://example.com/doc.pdf",
            )
        with self.assertRaises(InvalidRequestError):
            client.parse_document(
                model=Model.PP_OCRV5,
                file_url="http://example.com/doc.pdf",
            )

        client._submit.assert_not_called()

    def test_new_error_types_are_sdk_errors(self):
        self.assertTrue(issubclass(ResponseFormatError, PaddleOCRAPIError))
        self.assertTrue(issubclass(ResultParseError, PaddleOCRAPIError))
        self.assertTrue(issubclass(RequestTimeoutError, PaddleOCRAPIError))
        self.assertTrue(issubclass(PollTimeoutError, PaddleOCRAPIError))
        self.assertTrue(issubclass(FileNotFoundError, PaddleOCRAPIError))

    def test_raise_for_response_accepts_202(self):
        _raise_for_response(MockResponse({"accepted": True}, status_code=202))

    def test_raise_for_response_maps_401_and_403_to_auth_error(self):
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                with self.assertRaises(AuthError):
                    _raise_for_response(
                        MockResponse(
                            {"message": "bad token"},
                            status_code=status_code,
                        )
                    )

    def test_raise_for_response_maps_400_to_invalid_request_error(self):
        with self.assertRaises(InvalidRequestError):
            _raise_for_response(
                MockResponse({"message": "invalid input"}, status_code=400)
            )

    def test_raise_for_response_maps_other_non_2xx_to_api_error(self):
        with self.assertRaises(APIError) as ctx:
            _raise_for_response(
                MockResponse({"message": "server failed"}, status_code=500)
            )
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("paddleocr._api_client._http.requests.Session")
    def test_submit_missing_data_job_id_raises_response_format_error(
        self, mock_session_cls
    ):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse({"data": {}})

        client = APIClient()
        with self.assertRaises(ResponseFormatError):
            client.submit_ocr(file_url="http://example.com/test.pdf")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_get_job_status_missing_or_invalid_state_raises_response_format_error(
        self, mock_session_cls
    ):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = [
            MockResponse({"data": {}}),
            MockResponse({"data": {"state": "surprising"}}),
        ]

        client = APIClient()
        with self.assertRaises(ResponseFormatError):
            client.get_status("job-missing")
        with self.assertRaises(ResponseFormatError):
            client.get_status("job-unknown")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_requests_timeout_maps_to_request_timeout_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.side_effect = requests.Timeout("too slow")

        client = APIClient(request_timeout=1.0, poll_timeout=2.0)
        with self.assertRaises(RequestTimeoutError):
            client.submit_ocr(file_url="http://example.com/test.pdf")

    @patch("paddleocr._api_client._http.requests.get")
    def test_fetch_jsonl_uses_get_without_authorization_headers(self, mock_get):
        line = json.dumps({"result": {"ocrResults": []}})
        mock_resp = MagicMock()
        mock_resp.text = line
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        hc = HTTPClient(
            "secret-token",
            "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
            30.0,
        )

        hc.fetch_jsonl("https://bucket.example/results/out.jsonl")
        mock_get.assert_called_once_with(
            "https://bucket.example/results/out.jsonl", timeout=30.0
        )
        kwargs = mock_get.call_args.kwargs
        self.assertNotIn("headers", kwargs)

    @patch("paddleocr._api_client._http.requests.get")
    def test_fetch_jsonl_malformed_line_raises_result_parse_error_without_auth(
        self, mock_get
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "{bad json"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        hc = HTTPClient(
            "secret-token",
            "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
            30.0,
        )

        with self.assertRaises(ResultParseError):
            hc.fetch_jsonl("https://bucket.example/results/out.jsonl")
        kwargs = mock_get.call_args.kwargs
        self.assertNotIn("headers", kwargs)

    def test_poll_done_without_result_url_raises_response_format_error(self):
        http = MagicMock()
        http.get_job_status.return_value = {"state": "done"}
        poller = Poller(http, initial_interval=0.0, max_wait_time=1.0)

        with self.assertRaises(ResponseFormatError):
            poller.poll_until_done("job-no-url")

    def test_poll_unknown_state_raises_response_format_error(self):
        http = MagicMock()
        http.get_job_status.return_value = {"state": "paused"}
        poller = Poller(http, initial_interval=0.0, max_wait_time=1.0)

        with self.assertRaises(ResponseFormatError):
            poller.poll_until_done("job-paused")

    def test_poll_timeout_raises_poll_timeout_error(self):
        http = MagicMock()
        http.get_job_status.return_value = {"state": "pending"}
        poller = Poller(http, initial_interval=0.0, max_wait_time=0.0)

        with self.assertRaises(PollTimeoutError):
            poller.poll_until_done("job-slow")

    def test_poll_timeout_does_not_poll_after_deadline(self):
        http = MagicMock()
        http.get_job_status.return_value = {"state": "pending"}
        monotonic_values = [0.0, 0.0, 0.0, 1.0]
        poller = Poller(http, initial_interval=10.0, max_wait_time=1.0)

        with patch(
            "paddleocr._api_client._poller.time.monotonic",
            side_effect=monotonic_values,
        ):
            with patch("paddleocr._api_client._poller.time.sleep") as sleep:
                with self.assertRaises(PollTimeoutError):
                    poller.poll_until_done("job-deadline")

        self.assertEqual(http.get_job_status.call_count, 1)
        self.assertLessEqual(http.get_job_status.call_args.kwargs["timeout"], 1.0)
        sleep.assert_called_once_with(1.0)

    def test_poll_status_call_timeout_is_capped_to_remaining_deadline(self):
        http = MagicMock()
        http.get_job_status.return_value = {"state": "pending"}
        monotonic_values = [0.0, 0.75, 0.75, 1.0]
        poller = Poller(http, initial_interval=1.0, max_wait_time=1.0)

        with patch(
            "paddleocr._api_client._poller.time.monotonic",
            side_effect=monotonic_values,
        ):
            with patch("paddleocr._api_client._poller.time.sleep"):
                with self.assertRaises(PollTimeoutError):
                    poller.poll_until_done("job-near-deadline")

        self.assertEqual(http.get_job_status.call_args.kwargs["timeout"], 0.25)

    def test_malformed_ocr_jsonl_raises_result_parse_error(self):
        with self.assertRaises(ResultParseError):
            parse_ocr_result(
                "job-ocr", [{"result": {"ocrResults": [{"ocrImage": "x"}]}}]
            )

    def test_ocr_result_allows_missing_ocr_image(self):
        result = parse_ocr_result(
            "job-ocr",
            [{"result": {"ocrResults": [{"prunedResult": {"text": "hello"}}]}}],
        )

        self.assertEqual(result.pages[0].pruned_result, {"text": "hello"})
        self.assertIsNone(result.pages[0].ocr_image_url)

    def test_malformed_doc_parsing_jsonl_raises_result_parse_error(self):
        with self.assertRaises(ResultParseError):
            parse_doc_parsing_result(
                "job-doc", [{"result": {"layoutParsingResults": [{"markdown": {}}]}}]
            )

    @patch("paddleocr._api_client._http.requests.Session")
    def test_missing_local_file_raises_sdk_file_not_found(self, mock_session_cls):
        mock_session_cls.return_value = MagicMock()
        client = APIClient()

        with self.assertRaises(FileNotFoundError):
            client.submit_ocr(file_path="/path/that/does/not/exist.pdf")

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_downloads_single_url(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"image bytes"
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = APIClient(request_timeout=12.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")

            saved_path = client.save_resource("http://example.com/ocr.png", destination)

            self.assertEqual(saved_path, destination)
            with open(destination, "rb") as f:
                self.assertEqual(f.read(), b"image bytes")
        mock_get.assert_called_once_with("http://example.com/ocr.png", timeout=12.0)

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_refuses_overwrite_by_default(self, mock_get):
        client = APIClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")
            with open(destination, "wb") as f:
                f.write(b"existing")

            with self.assertRaises(InvalidRequestError):
                client.save_resource("http://example.com/ocr.png", destination)

        mock_get.assert_not_called()

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_overwrite_replaces_existing_file(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"new"
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = APIClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")
            with open(destination, "wb") as f:
                f.write(b"old")

            client.save_resource(
                "http://example.com/ocr.png",
                destination,
                overwrite=True,
            )

            with open(destination, "rb") as f:
                self.assertEqual(f.read(), b"new")

    def test_save_resource_missing_parent_raises_sdk_file_not_found(self):
        client = APIClient()

        with self.assertRaises(FileNotFoundError):
            client.save_resource(
                "http://example.com/ocr.png",
                "/path/that/does/not/exist/ocr.png",
            )

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_saves_result_resources_and_returns_summary(self, mock_get):
        def response_for(url, timeout):
            response = MagicMock()
            response.content = url.encode("utf-8")
            response.status_code = 200
            response.raise_for_status = MagicMock()
            return response

        mock_get.side_effect = response_for
        result = DocParsingResult(
            job_id="job-doc",
            pages=[
                DocParsingPage(
                    markdown_text="# Title",
                    markdown_images={"fig.png": "http://example.com/fig.png"},
                    output_images={"page": "http://example.com/page.png"},
                )
            ],
        )

        client = APIClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = client.save_resource(result, tmpdir)

            self.assertEqual(len(summary.saved_paths), 2)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "fig.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "page")))

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_saves_ocr_result_resources(self, mock_get):
        def response_for(url, timeout):
            response = MagicMock()
            response.content = url.encode("utf-8")
            response.status_code = 200
            return response

        mock_get.side_effect = response_for
        result = OCRResult(
            job_id="job-ocr",
            pages=[
                OCRPage(
                    pruned_result={"text": "hello"},
                    ocr_image_url="http://example.com/ocr-page.png",
                )
            ],
        )

        client = APIClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = client.save_resource(result, tmpdir)

            self.assertEqual(
                summary.saved_paths,
                [os.path.join(tmpdir, "ocr-page.png")],
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "ocr-page.png")))

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_prefers_document_resource_map_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"image"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        result = DocParsingResult(
            job_id="job-doc",
            pages=[
                DocParsingPage(
                    markdown_text="# Title",
                    markdown_images={"fig.png": "http://example.com/opaque/download"},
                )
            ],
        )

        client = APIClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = client.save_resource(result, tmpdir)

            self.assertEqual(summary.saved_paths, [os.path.join(tmpdir, "fig.png")])
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "fig.png")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "download")))

    def test_save_resource_rejects_unsafe_result_resource_names(self):
        client = APIClient()
        result = DocParsingResult(
            job_id="job-doc",
            pages=[
                DocParsingPage(
                    markdown_text="# Title",
                    markdown_images={"../fig.png": "http://example.com/fig.png"},
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(InvalidRequestError):
                client.save_resource(result, tmpdir)

        result.pages[0].markdown_images = {
            "nested/fig.png": "http://example.com/fig.png"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(InvalidRequestError):
                client.save_resource(result, tmpdir)

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_timeout_maps_to_request_timeout_error(self, mock_get):
        mock_get.side_effect = requests.Timeout("too slow")
        client = APIClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")
            with self.assertRaises(RequestTimeoutError):
                client.save_resource("http://example.com/ocr.png", destination)

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_connection_error_maps_to_network_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("connection failed")
        client = APIClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")
            with self.assertRaises(NetworkError):
                client.save_resource("http://example.com/ocr.png", destination)

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_request_exception_maps_to_network_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("transport failed")
        client = APIClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")
            with self.assertRaises(NetworkError):
                client.save_resource("http://example.com/ocr.png", destination)

    @patch("paddleocr._api_client.client.requests.get")
    def test_save_resource_non_2xx_maps_to_api_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "service unavailable"
        mock_response.json.side_effect = ValueError("not json")
        mock_get.return_value = mock_response
        client = APIClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "ocr.png")
            with self.assertRaises(APIError) as ctx:
                client.save_resource("http://example.com/ocr.png", destination)

        self.assertEqual(ctx.exception.status_code, 503)

    def test_public_api_exports_result_and_progress_types(self):
        import paddleocr
        import paddleocr._api_client as api_client

        for name in [
            "APIClient",
            "AsyncAPIClient",
            "OCRResult",
            "OCRPage",
            "DocParsingResult",
            "DocParsingPage",
            "Job",
            "JobStatus",
            "Progress",
        ]:
            self.assertIn(name, paddleocr.__all__)
            self.assertIn(name, api_client.__all__)

        self.assertIs(paddleocr.OCRResult, OCRResult)
        self.assertIs(paddleocr.OCRPage, OCRPage)
        self.assertIs(paddleocr.DocParsingResult, DocParsingResult)
        self.assertIs(paddleocr.DocParsingPage, DocParsingPage)
        self.assertIs(paddleocr.Job, Job)
        self.assertIs(paddleocr.Progress, Progress)
        self.assertNotIn("TimeoutError", paddleocr.__all__)
        self.assertNotIn("TimeoutError", api_client.__all__)

    def test_api_cli_help_includes_request_and_poll_timeout_options(self):
        import argparse

        parser = argparse.ArgumentParser(prog="paddleocr")
        subparsers = parser.add_subparsers(dest="subcommand")
        register_api_command(subparsers)

        help_text = parser.format_help()

        self.assertIn("api", help_text)
        api_help = parser._subparsers._group_actions[0].choices["api"].format_help()
        self.assertIn("--request_timeout", api_help)
        self.assertIn("--poll_timeout", api_help)

    @patch("paddleocr._api_client.cli.APIClient")
    def test_api_cli_ocr_url_flow_uses_timeout_options(self, mock_client_cls):
        client = MagicMock()
        client.ocr.return_value = OCRResult(
            job_id="job-cli",
            pages=[OCRPage(pruned_result={"text": "cli"}, ocr_image_url="http://img")],
        )
        mock_client_cls.return_value = client
        args = Namespace(
            token="token",
            request_timeout=7.0,
            poll_timeout=11.0,
            model_type="ocr",
            model=None,
            file_url="http://example.com/test.png",
            file_path=None,
            output=None,
            page_ranges=None,
            batch_id=None,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            use_chart_recognition=None,
        )

        with patch("builtins.print") as mock_print:
            _execute_api(args)

        mock_client_cls.assert_called_once_with(
            token="token",
            request_timeout=7.0,
            poll_timeout=11.0,
        )
        client.ocr.assert_called_once()
        self.assertIn("job-cli", mock_print.call_args.args[0])

    @patch("paddleocr._api_client.cli.APIClient")
    def test_api_cli_ocr_validation_uses_model_classification_helper(
        self, mock_client_cls
    ):
        client = MagicMock()
        client.ocr.return_value = OCRResult(job_id="job-cli", pages=[])
        mock_client_cls.return_value = client
        args = Namespace(
            token="token",
            request_timeout=None,
            poll_timeout=None,
            model_type="ocr",
            model=Model.PP_STRUCTURE_V3.value,
            file_url="http://example.com/test.png",
            file_path=None,
            output=None,
            page_ranges=None,
            batch_id=None,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            use_chart_recognition=None,
        )

        def fake_is_ocr_model(model):
            return model == Model.PP_STRUCTURE_V3

        with patch("paddleocr._api_client.cli.is_ocr_model", fake_is_ocr_model):
            with patch("builtins.print"):
                _execute_api(args)

        client.ocr.assert_called_once()
        self.assertEqual(client.ocr.call_args.kwargs["model"], Model.PP_STRUCTURE_V3)

    @patch("paddleocr._api_client.cli.APIClient")
    def test_api_cli_document_parsing_rejection_uses_ocr_helper(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        args = Namespace(
            token="token",
            request_timeout=None,
            poll_timeout=None,
            model_type="document_parsing",
            model=Model.PP_STRUCTURE_V3.value,
            file_url="http://example.com/test.png",
            file_path=None,
            output=None,
            page_ranges=None,
            batch_id=None,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            use_chart_recognition=None,
        )

        def fake_is_ocr_model(model):
            return model == Model.PP_STRUCTURE_V3

        with patch("paddleocr._api_client.cli.is_ocr_model", fake_is_ocr_model):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit) as ctx:
                    _execute_api(args)

        self.assertEqual(ctx.exception.code, 2)
        mock_client_cls.assert_not_called()
        client.parse_document.assert_not_called()

    @patch("paddleocr._api_client.cli.APIClient")
    def test_api_cli_rejects_ocr_model_for_document_parsing(self, mock_client_cls):
        client = MagicMock()
        mock_client_cls.return_value = client
        args = Namespace(
            token="token",
            request_timeout=None,
            poll_timeout=None,
            model_type="document_parsing",
            model=Model.PP_OCRV5.value,
            file_url="http://example.com/test.png",
            file_path=None,
            output=None,
            page_ranges=None,
            batch_id=None,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            use_chart_recognition=None,
        )

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                _execute_api(args)

        self.assertEqual(ctx.exception.code, 2)
        mock_client_cls.assert_not_called()
        client.parse_document.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_api_cli_rejects_invalid_document_model_before_auth(self):
        args = Namespace(
            token=None,
            request_timeout=None,
            poll_timeout=None,
            model_type="document_parsing",
            model=Model.PP_OCRV5.value,
            file_url="http://example.com/test.png",
            file_path=None,
            output=None,
            page_ranges=None,
            batch_id=None,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            use_chart_recognition=None,
        )

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                _execute_api(args)

        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
