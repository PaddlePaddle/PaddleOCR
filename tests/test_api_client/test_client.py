import json
import os
import unittest
from unittest.mock import MagicMock, patch

from paddleocr import (
    PaddleOCRClient,
    Model,
    OCROptions,
    PaddleOCRVLOptions,
    PPStructureV3Options,
)
from paddleocr._api_client._http import HTTPClient
from paddleocr._api_client.errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    JobFailedError,
    ResponseFormatError,
)


class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self._json = json_data
        self.status_code = status_code
        self.text = text or json.dumps(json_data)

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


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
            PaddleOCRClient()

    def test_old_client_name_is_not_exported(self):
        import paddleocr

        self.assertFalse(hasattr(paddleocr, "APIClient"))

    def test_no_file_raises(self):
        client = PaddleOCRClient()
        with self.assertRaises(InvalidRequestError):
            client.ocr()

    def test_both_file_raises(self):
        client = PaddleOCRClient()
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
            client = PaddleOCRClient()
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
    def test_doc_parsing_url(self, mock_session_cls):
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
            client = PaddleOCRClient()
            with patch("paddleocr._api_client._poller.time.sleep"):
                result = client.parse_document(
                    model=Model.PP_STRUCTURE_V3,
                    file_url="http://example.com/doc.pdf",
                    options=PPStructureV3Options(use_chart_recognition=True),
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

        client = PaddleOCRClient()
        with patch("paddleocr._api_client._poller.time.sleep"):
            with self.assertRaises(JobFailedError) as ctx:
                client.ocr(file_url="http://example.com/bad.pdf")
            self.assertIn("File corrupted", str(ctx.exception))

    def test_context_manager(self):
        with PaddleOCRClient() as client:
            self.assertIsNotNone(client)

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

        opts2 = PPStructureV3Options(use_chart_recognition=True)
        payload2 = opts2.to_payload()
        self.assertEqual(
            payload2,
            {
                "useChartRecognition": True,
            },
        )

        vl_opts = PaddleOCRVLOptions(
            prompt_label="table",
            temperature=0.1,
            top_p=0.8,
            min_pixels=128,
            max_pixels=2048,
            restructure_pages=True,
        )
        self.assertEqual(
            vl_opts.to_payload(),
            {
                "promptLabel": "table",
                "temperature": 0.1,
                "topP": 0.8,
                "minPixels": 128,
                "maxPixels": 2048,
                "restructurePages": True,
            },
        )

    def test_vl_options_validate_ranges(self):
        with self.assertRaises(InvalidRequestError):
            PaddleOCRVLOptions(top_p=2).to_payload()
        with self.assertRaises(InvalidRequestError):
            PaddleOCRVLOptions(min_pixels=100, max_pixels=10).to_payload()

    @patch("paddleocr._api_client._http.requests.Session")
    def test_submit_includes_top_level_job_parameters(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse({"data": {"jobId": "job-789"}})

        client = PaddleOCRClient()
        job = client.submit_ocr(
            file_url="http://example.com/test.pdf",
            page_ranges="1,3-4",
            batch_id="batch-1",
        )

        self.assertEqual(job.job_id, "job-789")
        _, kwargs = mock_session.post.call_args
        self.assertEqual(kwargs["json"]["pageRanges"], "1,3-4")
        self.assertEqual(kwargs["json"]["batchId"], "batch-1")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_get_batch_status_uses_official_batch_endpoint(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.return_value = MockResponse(
            {
                "code": 0,
                "msg": "Success",
                "data": {
                    "batchId": "batch-1",
                    "extractResult": [
                        {
                            "jobId": "job-1",
                            "state": "done",
                            "resultUrl": {"jsonUrl": "https://bucket/result.jsonl"},
                            "extractProgress": {"totalPages": 2, "extractedPages": 2},
                        }
                    ],
                },
            }
        )

        client = PaddleOCRClient()
        batch = client.get_batch_status("batch-1")

        mock_session.get.assert_called_once()
        self.assertTrue(mock_session.get.call_args.args[0].endswith("/batch/batch-1"))
        self.assertEqual(batch.batch_id, "batch-1")
        self.assertEqual(batch.jobs[0].job_id, "job-1")
        self.assertEqual(batch.jobs[0].state, "done")

    @patch("paddleocr._api_client._http.requests.Session")
    def test_api_error_message_uses_official_msg_field(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse(
            {"code": 10007, "msg": "模型参数错误"},
            status_code=400,
        )

        client = PaddleOCRClient()
        with self.assertRaises(InvalidRequestError) as ctx:
            client.submit_ocr(file_url="https://example.com/file.pdf")
        self.assertIn("模型参数错误", str(ctx.exception))

    @patch("paddleocr._api_client._http.requests.Session")
    def test_successful_nonzero_code_is_api_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse(
            {"code": 10010, "msg": "任务提交队列已满", "data": {}},
            status_code=200,
        )

        client = PaddleOCRClient()
        with self.assertRaises(APIError) as ctx:
            client.submit_ocr(file_url="https://example.com/file.pdf")
        self.assertIn("任务提交队列已满", str(ctx.exception))

    @patch("paddleocr._api_client._http.requests.Session")
    def test_missing_job_id_is_response_format_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = MockResponse(
            {"code": 0, "msg": "Success", "data": {}},
            status_code=200,
        )

        client = PaddleOCRClient()
        with self.assertRaises(ResponseFormatError):
            client.submit_ocr(file_url="https://example.com/file.pdf")

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


if __name__ == "__main__":
    unittest.main()
