---
comments: true
---

# PaddleOCR official API Python SDK

The Python SDK calls the PaddleOCR official API through `PaddleOCRClient` and `AsyncPaddleOCRClient` in the `paddleocr` package. It submits OCR or document parsing jobs to hosted services. It does not run local inference or load local models.

## Install And Authenticate

First install the Python package by following [Install paddleocr](../../../installation.en.md#11-install-paddleocr). After installing the core `paddleocr` package, this feature is available out of the box — no additional dependency groups are required.

First obtain an access token from the [AI Studio Access Token page](https://aistudio.baidu.com/account/accessToken).

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

`PaddleOCRClient()` reads `PADDLEOCR_ACCESS_TOKEN` by default and also accepts `PaddleOCRClient(token="...")`. Missing credentials raise `AuthError`.

## Quick Start

```python
from paddleocr import PaddleOCRClient, Model

client = PaddleOCRClient()
result = client.ocr(
    file_url="https://example.com/invoice.pdf",
    model=Model.PP_OCRV5,
)
print(result.job_id, len(result.pages))
client.close()
```

Use `file_path` for a local file. Pass exactly one of `file_url` or `file_path`.

## Public API

Common Python public methods include:

- `ocr(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `parse_document(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `submit_ocr(...)`: submit only an OCR job and return a job object.
- `submit_document_parsing(...)`: submit only a document parsing job and return a job object.
- `get_status(job_id)`: perform one non-blocking status request without waiting for completion.
- `wait_ocr_result(job)`: wait for an OCR job and parse its result.
- `wait_document_parsing_result(job)`: wait for a document parsing job and parse its result.
- `save_resource(resource_url, destination)`: save one resource URL.
- `save_ocr_result_resources(result, destination)`: save resources referenced by an OCR result object.
- `save_document_parsing_result_resources(result, destination)`: save resources referenced by a document parsing result object.

`AsyncPaddleOCRClient` exposes async versions of these job operations and resource-saving methods.

## Timeouts

```python
client = PaddleOCRClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` limits one HTTP request, including submit, status, and result-resource downloads. `poll_timeout` limits the total wait time for `ocr`, `parse_document`, `wait_ocr_result`, and `wait_document_parsing_result`.

## Errors And Resource Saving

All SDK errors inherit from `PaddleOCRAPIError`. Common typed errors include `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

Use `save_resource` for one resource URL. To save all resources referenced by a result object, use `save_ocr_result_resources` or `save_document_parsing_result_resources`.

## Batch Status

When submitting jobs, pass `batch_id`. Later, use `client.get_batch_status("batch-id")` to inspect each job's state, progress, and result URL in that batch.

## Choose Models

| Task | Interfaces | Default model | Supported models | Option type |
| --- | --- | --- | --- | --- |
| OCR | `ocr`, `submit_ocr`, `wait_ocr_result` | `Model.PP_OCRV5` | `Model.PP_OCRV5` | `OCROptions` |
| Document parsing | `parse_document`, `submit_document_parsing`, `wait_document_parsing_result` | `Model.PADDLE_OCR_VL_16` | `Model.PP_STRUCTURE_V3`, `Model.PADDLE_OCR_VL`, `Model.PADDLE_OCR_VL_15`, `Model.PADDLE_OCR_VL_16` | Use `PPStructureV3Options` with `PP-StructureV3`, and `PaddleOCRVLOptions` with PaddleOCR-VL models. |
