import json
import os
import tempfile
import unittest
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

from paddleocr import AsyncAPIClient, Job, Model
from paddleocr._api_client.errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    NetworkError,
    PollTimeoutError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
)
from paddleocr._api_client.models import is_document_parsing_model, is_ocr_model
from paddleocr._api_client.results import DocParsingResult, OCRResult


class AsyncResponse:
    def __init__(self, json_data=None, status=200, text=""):
        self._json = json_data
        self.status = status
        self._text = text or json.dumps(json_data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class TestAsyncAPIClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        os.environ["PADDLEOCR_ACCESS_TOKEN"] = "test-token"

    def tearDown(self):
        if "PADDLEOCR_ACCESS_TOKEN" in os.environ:
            del os.environ["PADDLEOCR_ACCESS_TOKEN"]

    async def test_async_client_exposes_only_contract_public_names(self):
        client = AsyncAPIClient()

        self.assertTrue(hasattr(client, "get_status"))
        self.assertFalse(hasattr(client, "get_result"))
        self.assertFalse(hasattr(client, "wait_for_result"))
        self.assertFalse(hasattr(client, "doc_parsing"))
        self.assertFalse(hasattr(client, "submit_doc_parsing"))
        self.assertTrue(hasattr(client, "parse_document"))
        self.assertTrue(hasattr(client, "submit_document_parsing"))
        self.assertFalse(hasattr(client, "wait_for_ocr_result"))
        self.assertTrue(hasattr(client, "wait_ocr_result"))

    async def test_submit_methods_return_job_metadata(self):
        client = AsyncAPIClient()
        client._submit = AsyncMock(side_effect=["ocr-job", "doc-job"])

        ocr_job = await client.submit_ocr(file_url="http://example.com/ocr.pdf")
        doc_job = await client.submit_document_parsing(
            model=Model.PADDLE_OCR_VL,
            file_url="http://example.com/doc.pdf",
        )

        self.assertEqual(ocr_job.job_id, "ocr-job")
        self.assertEqual(ocr_job.model, Model.PP_OCRV5.value)
        self.assertEqual(ocr_job.task, "ocr")
        self.assertEqual(doc_job.job_id, "doc-job")
        self.assertEqual(doc_job.model, Model.PADDLE_OCR_VL.value)
        self.assertEqual(doc_job.task, "document_parsing")

    async def test_submit_ocr_propagates_explicit_model(self):
        client = AsyncAPIClient()
        client._submit = AsyncMock(return_value="ocr-job")

        job = await client.submit_ocr(
            model=Model.PP_OCRV5,
            file_url="http://example.com/ocr.pdf",
        )

        self.assertEqual(job.model, Model.PP_OCRV5.value)
        client._submit.assert_awaited_once()
        self.assertEqual(client._submit.await_args.args[0], Model.PP_OCRV5)

    async def test_model_helpers_classify_current_ocr_model(self):
        self.assertTrue(is_ocr_model(Model.PP_OCRV5))
        self.assertTrue(is_ocr_model(Model.PP_OCRV5.value))
        self.assertFalse(is_ocr_model(Model.PP_STRUCTURE_V3))
        self.assertFalse(is_ocr_model("future-unknown-model"))
        self.assertTrue(is_document_parsing_model(Model.PADDLE_OCR_VL))
        self.assertFalse(is_document_parsing_model(Model.PP_OCRV5))

    async def test_submit_ocr_rejects_non_ocr_model(self):
        client = AsyncAPIClient()
        client._submit = AsyncMock()

        with self.assertRaises(InvalidRequestError):
            await client.submit_ocr(
                model=Model.PP_STRUCTURE_V3,
                file_url="http://example.com/ocr.pdf",
            )

        client._submit.assert_not_awaited()

    async def test_job_task_public_literal_uses_contract_names(self):
        task_values = set(get_args(Job.__annotations__["task"]))

        self.assertEqual(task_values, {"ocr", "document_parsing"})
        self.assertNotIn("doc_parsing", task_values)

    async def test_document_parsing_rejects_ocr_model(self):
        client = AsyncAPIClient()
        client._submit = AsyncMock()

        with self.assertRaises(InvalidRequestError):
            await client.submit_document_parsing(
                model=Model.PP_OCRV5,
                file_url="http://example.com/doc.pdf",
            )
        with self.assertRaises(InvalidRequestError):
            await client.submit_document_parsing(
                model=Model.PP_OCRV5.value,
                file_url="http://example.com/doc.pdf",
            )
        with self.assertRaises(InvalidRequestError):
            await client.parse_document(
                model=Model.PP_OCRV5,
                file_url="http://example.com/doc.pdf",
            )

        client._submit.assert_not_awaited()

    async def test_wait_ocr_result_accepts_job_and_parses_ocr_result(self):
        client = AsyncAPIClient()
        client._poll_until_done = AsyncMock(
            return_value=[
                {
                    "result": {
                        "ocrResults": [
                            {
                                "prunedResult": {"text": "async ocr"},
                                "ocrImage": "http://img/async.png",
                            }
                        ]
                    }
                }
            ]
        )

        result = await client.wait_ocr_result(
            Job(job_id="job-ocr", model=Model.PP_OCRV5.value, task="ocr")
        )

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.job_id, "job-ocr")
        self.assertEqual(result.pages[0].pruned_result, {"text": "async ocr"})

    async def test_wait_document_parsing_result_accepts_job_and_parses_doc_result(self):
        client = AsyncAPIClient()
        client._poll_until_done = AsyncMock(
            return_value=[
                {
                    "result": {
                        "layoutParsingResults": [
                            {
                                "markdown": {
                                    "text": "# Async",
                                    "images": {"img.png": "http://img/async-doc.png"},
                                },
                                "outputImages": {"page": "http://out/async-page.png"},
                            }
                        ]
                    }
                }
            ]
        )

        result = await client.wait_document_parsing_result(
            Job(
                job_id="job-doc",
                model=Model.PP_STRUCTURE_V3.value,
                task="document_parsing",
            )
        )

        self.assertIsInstance(result, DocParsingResult)
        self.assertEqual(result.job_id, "job-doc")
        self.assertEqual(result.pages[0].markdown_text, "# Async")

    async def test_wait_document_parsing_result_rejects_unknown_model(self):
        client = AsyncAPIClient()
        client._poll_until_done = AsyncMock(
            return_value=[
                {
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "# Should not poll"}}
                        ]
                    }
                }
            ]
        )

        with self.assertRaises(InvalidRequestError):
            await client.wait_document_parsing_result(
                Job(
                    job_id="job-doc",
                    model="future-unknown-model",
                    task="document_parsing",
                )
            )
        client._poll_until_done.assert_not_awaited()

    async def test_typed_wait_methods_reject_mismatched_jobs(self):
        client = AsyncAPIClient()

        with self.assertRaises(InvalidRequestError):
            await client.wait_ocr_result(
                Job(
                    job_id="job-doc",
                    model=Model.PP_STRUCTURE_V3.value,
                    task="document_parsing",
                )
            )

        with self.assertRaises(InvalidRequestError):
            await client.wait_document_parsing_result(
                Job(job_id="job-ocr", model=Model.PP_OCRV5.value, task="ocr")
            )

    async def test_typed_wait_accepts_bare_job_id_and_uses_method_parser(self):
        client = AsyncAPIClient()
        client._poll_until_done = AsyncMock(
            return_value=[
                {
                    "result": {
                        "ocrResults": [
                            {
                                "prunedResult": {"text": "bare"},
                                "ocrImage": "http://img/bare.png",
                            }
                        ]
                    }
                }
            ]
        )

        result = await client.wait_ocr_result("job-string")

        self.assertEqual(result.job_id, "job-string")
        self.assertEqual(result.pages[0].pruned_result, {"text": "bare"})

    async def test_async_http_error_mapping(self):
        client = AsyncAPIClient()

        with self.assertRaises(AuthError):
            await client._raise_for_response(
                AsyncResponse({"message": "bad token"}, status=401)
            )
        with self.assertRaises(InvalidRequestError):
            await client._raise_for_response(
                AsyncResponse({"message": "bad input"}, status=400)
            )
        with self.assertRaises(APIError) as ctx:
            await client._raise_for_response(
                AsyncResponse({"message": "server failed"}, status=500)
            )
        self.assertEqual(ctx.exception.status_code, 500)

    async def test_async_poll_timeout_raises_poll_timeout_error(self):
        client = AsyncAPIClient(poll_timeout=0.0)
        client._get_job_status = AsyncMock(return_value={"state": "pending"})

        with self.assertRaises(PollTimeoutError):
            await client._poll_until_done("slow-job")

    async def test_async_poll_timeout_does_not_poll_after_deadline(self):
        client = AsyncAPIClient(poll_timeout=1.0)
        client._get_job_status = AsyncMock(return_value={"state": "pending"})
        monotonic_values = [0.0, 0.0, 0.0, 1.0]

        with patch(
            "paddleocr._api_client.async_client.time.monotonic",
            side_effect=monotonic_values,
        ):
            with patch("paddleocr._api_client.async_client.asyncio.sleep") as sleep:
                with self.assertRaises(PollTimeoutError):
                    await client._poll_until_done("slow-job")

        self.assertEqual(client._get_job_status.call_count, 1)
        self.assertLessEqual(client._get_job_status.call_args.kwargs["timeout"], 1.0)
        sleep.assert_awaited_once_with(1.0)

    async def test_async_poll_status_timeout_is_capped_to_remaining_deadline(self):
        client = AsyncAPIClient(poll_timeout=1.0)
        client._get_job_status = AsyncMock(return_value={"state": "pending"})
        monotonic_values = [0.0, 0.75, 0.75, 1.0]

        with patch(
            "paddleocr._api_client.async_client.time.monotonic",
            side_effect=monotonic_values,
        ):
            with patch("paddleocr._api_client.async_client.asyncio.sleep"):
                with self.assertRaises(PollTimeoutError):
                    await client._poll_until_done("slow-job")

        self.assertEqual(client._get_job_status.call_args.kwargs["timeout"], 0.25)

    async def test_async_malformed_jsonl_raises_result_parse_error(self):
        client = AsyncAPIClient()
        with self.assertRaises(ResultParseError):
            client._parse_jsonl_text("{bad json")

    async def test_async_response_json_preserves_timeout_and_client_errors(self):
        class TimeoutResponse:
            async def json(self):
                raise TimeoutError("body timed out")

        client = AsyncAPIClient()

        with self.assertRaises(TimeoutError):
            await client._response_json(TimeoutResponse())

    async def test_async_response_json_maps_value_error_to_response_format_error(self):
        class BadJsonResponse:
            async def json(self):
                raise ValueError("not json")

        client = AsyncAPIClient()

        with self.assertRaises(ResponseFormatError):
            await client._response_json(BadJsonResponse())

    async def test_async_response_json_maps_content_type_error_to_response_format_error(
        self,
    ):
        import aiohttp

        class NonJsonResponse:
            async def json(self):
                raise aiohttp.ContentTypeError(
                    request_info=None,
                    history=(),
                    message="unexpected mimetype",
                )

        client = AsyncAPIClient()

        with self.assertRaises(ResponseFormatError):
            await client._response_json(NonJsonResponse())

    async def test_async_done_without_result_url_raises_response_format_error(self):
        client = AsyncAPIClient()
        client._get_job_status = AsyncMock(return_value={"state": "done"})

        with self.assertRaises(ResponseFormatError):
            await client._poll_until_done("job-no-result")

    async def test_async_unknown_state_raises_response_format_error(self):
        client = AsyncAPIClient()
        client._get_job_status = AsyncMock(return_value={"state": "paused"})

        with self.assertRaises(ResponseFormatError):
            await client._poll_until_done("job-unknown")

    async def test_async_successful_submit_response_requires_job_id(self):
        client = AsyncAPIClient()
        client._session = MagicMock()
        client._session.post.return_value = AsyncResponse({"data": {}}, status=202)

        with self.assertRaises(ResponseFormatError):
            await client._submit_url(
                Model.PP_OCRV5.value,
                "http://example.com/test.pdf",
                {},
            )

    async def test_async_submit_file_streams_file_object(self):
        captured_file = None
        test_case = self

        class StreamingResponse(AsyncResponse):
            async def __aenter__(self):
                self_outer = await super().__aenter__()
                test_case.assertIsNotNone(captured_file)
                test_case.assertFalse(captured_file.closed)
                return self_outer

        class FormData:
            def add_field(self, name, value, **kwargs):
                nonlocal captured_file
                if name == "file":
                    captured_file = value

        client = AsyncAPIClient()
        client._session = MagicMock()
        client._session.post.return_value = StreamingResponse(
            {"data": {"jobId": "job-file"}},
            status=200,
        )

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"example")
            tmp.flush()
            with patch("aiohttp.FormData", return_value=FormData()):
                job_id = await client._submit_file(
                    Model.PP_OCRV5.value,
                    tmp.name,
                    {},
                )

        self.assertEqual(job_id, "job-file")
        self.assertTrue(captured_file.closed)


if __name__ == "__main__":
    unittest.main()
