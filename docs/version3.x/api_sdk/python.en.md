---
comments: true
---

# PaddleOCR official API Python SDK

The Python SDK calls the PaddleOCR official API through `APIClient` and `AsyncAPIClient` in the `paddleocr` package. It submits OCR or document parsing jobs to hosted services. It does not run local inference or load local models.

## Install And Authenticate

During local development, install PaddleOCR from this source tree:

```bash
python -m pip install -e .
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

`APIClient()` reads `PADDLEOCR_ACCESS_TOKEN` by default and also accepts `APIClient(token="...")`. Missing credentials raise `AuthError`.

## Quick Start

```python
from paddleocr import APIClient, Model

client = APIClient()
result = client.ocr(
    file_url="https://example.com/invoice.pdf",
    model=Model.PP_OCRV5,
)
print(result.job_id, len(result.pages))
client.close()
```

Use `file_path` for a local file. Pass exactly one of `file_url` or `file_path`.

Document parsing example:

```python
from paddleocr import APIClient, DocParsingOptions, Model

client = APIClient()
result = client.parse_document(
    model=Model.PADDLE_OCR_VL_15,
    file_path="./report.pdf",
    options=DocParsingOptions(use_chart_recognition=True),
)
print(result.job_id, len(result.pages))
for page in result.pages:
    print(page.markdown_text)
client.close()
```

## Public API

The final Python public methods are:

- `ocr(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `parse_document(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `submit_ocr(...)`: submit only an OCR job and return a job object.
- `submit_document_parsing(...)`: submit only a document parsing job and return a job object.
- `get_status(job_id)`: perform one non-blocking status request without waiting for completion.
- `wait_ocr_result(job)`: wait for an OCR job and parse its result.
- `wait_document_parsing_result(job)`: wait for a document parsing job and parse its result.
- `save_resource(resource, destination, overwrite=False)`: save one resource URL or resources referenced by a result object.

`AsyncAPIClient` exposes async versions of job operations: `ocr`, `parse_document`, `submit_ocr`, `submit_document_parsing`, `get_status`, `wait_ocr_result`, `wait_document_parsing_result`, and `close`. It does not currently provide a `save_resource` coroutine method.

## Timeouts

```python
client = APIClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` limits one HTTP request, including submit, status, and result-resource downloads. `poll_timeout` limits the total wait time for `ocr`, `parse_document`, `wait_ocr_result`, and `wait_document_parsing_result`.

## Model Extensibility

The OCR `model` is optional and defaults to `Model.PP_OCRV5`. PP-OCRv5 is the only OCR model exposed by the current PaddleOCR official API release.

Document parsing `model` is optional and defaults to `Model.PADDLE_OCR_VL_15`. Supported document parsing models include `Model.PP_STRUCTURE_V3`, `Model.PADDLE_OCR_VL`, and `Model.PADDLE_OCR_VL_15`. The SDK validates model categories through `is_ocr_model` and `is_document_parsing_model`, so future models can be added centrally.

## Errors And Resource Saving

All SDK errors inherit from `PaddleOCRAPIError`. Common typed errors include `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

Resource saving is currently provided by the sync `APIClient.save_resource` method. Async users can use `APIClient.save_resource` after obtaining result resource URLs, or implement custom async downloads if they need end-to-end async I/O. Do not use `await async_client.save_resource(...)`; that method does not currently exist.

`APIClient.save_resource` can save one resource URL or all resources referenced by an `OCRResult` or `DocParsingResult`. It does not overwrite existing files by default and does not send PaddleOCR official API authorization headers to result resource URLs.

See the [api_sdk/PYTHON.md](../../../api_sdk/PYTHON.md) source-adjacent reference.
