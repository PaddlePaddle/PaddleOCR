---
comments: true
---

# PaddleOCR official API Python SDK

The Python SDK calls the PaddleOCR official API through `PaddleOCRClient` and `AsyncPaddleOCRClient` in the `paddleocr` package. It submits OCR or document parsing jobs to hosted services. It does not run local inference or load local models.

## Install And Authenticate

During local development, install PaddleOCR from this source tree:

```bash
python -m pip install -e .
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
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

The final Python public methods are:

- `ocr(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `parse_document(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `submit_ocr(...)`: submit only an OCR job and return a job object.
- `submit_document_parsing(...)`: submit only a document parsing job and return a job object.
- `get_status(job_id)`: perform one non-blocking status request without waiting for completion.
- `wait_ocr_result(job)`: wait for an OCR job and parse its result.
- `wait_document_parsing_result(job)`: wait for a document parsing job and parse its result.

`AsyncPaddleOCRClient` exposes async versions of job operations: `ocr`, `parse_document`, `submit_ocr`, `submit_document_parsing`, `get_status`, `get_batch_status`, `wait_ocr_result`, `wait_document_parsing_result`, and `close`.

## Timeouts

```python
client = PaddleOCRClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` limits one HTTP request, including submit, status, and result-resource downloads. `poll_timeout` limits the total wait time for `ocr`, `parse_document`, `wait_ocr_result`, and `wait_document_parsing_result`.

## Model Extensibility

The OCR `model` is optional and defaults to `Model.PP_OCRV5`. PP-OCRv5 is the only OCR model exposed by the current PaddleOCR official API release. Document parsing `model` is optional and defaults to `Model.PADDLE_OCR_VL_16`. The SDK validates model categories through `is_ocr_model` and `is_document_parsing_model`, so future models can be added centrally.

## Errors And Resource Saving

All SDK errors inherit from `PaddleOCRAPIError`. Common typed errors include `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.


See `api_sdk/PYTHON.md` for the source-adjacent reference.


## Batch Status

When submitting jobs, pass `batch_id`. Later, use `client.get_batch_status("batch-id")` to inspect each job's state, progress, and result URL in that batch.

## Document Parsing Option Types

Use `PPStructureV3Options` with `PP-StructureV3`, and `PaddleOCRVLOptions` with `PaddleOCR-VL`, `PaddleOCR-VL-1.5`, and `PaddleOCR-VL-1.6`. This avoids accidentally sending VL-only parameters such as `prompt_label`, `temperature`, `top_p`, `min_pixels`, and `restructure_pages` to PP-StructureV3.
