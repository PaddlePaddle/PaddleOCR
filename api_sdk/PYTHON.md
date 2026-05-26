# PaddleOCR Python API SDK

The Python API client calls the PaddleOCR official API. It submits OCR or document
parsing jobs to hosted services and returns typed result objects. It does not run
local PaddleOCR inference or load local models.

This file is a source-adjacent reference for Python SDK maintenance. The official
user docs live in `docs/version3.x/api_sdk/python.md` and
`docs/version3.x/api_sdk/python.en.md`.

## Install And Import

Install PaddleOCR from this worktree during development:

```bash
python -m pip install -e .
```

```python
from paddleocr import (
    APIClient,
    AsyncAPIClient,
    DocParsingOptions,
    Model,
    OCROptions,
)
```

## Authentication

Set `PADDLEOCR_ACCESS_TOKEN` or pass `token` explicitly:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

```python
client = APIClient(token="your-api-token")
```

`APIClient` and `AsyncAPIClient` raise `AuthError` at construction when no token
is available.

## OCR From URL

```python
from paddleocr import APIClient, Model, OCROptions

client = APIClient()
result = client.ocr(
    file_url="https://example.com/invoice.pdf",
    model=Model.PP_OCRV5,
    options=OCROptions(use_textline_orientation=True),
)

for page in result.pages:
    print(page.pruned_result)
    print(page.ocr_image_url)
```

## OCR From File

```python
result = client.ocr(
    file_path="./invoice.pdf",
    page_ranges="1-3",
    options=OCROptions(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    ),
)
```

Pass exactly one of `file_url` or `file_path`.

`model` is optional for OCR and defaults to `Model.PP_OCRV5`. PP-OCRv5 is the
only OCR model supported by the current PaddleOCR official API release; model
validation is centralized in `is_ocr_model` and `is_document_parsing_model` so
future OCR models can be added without changing submit/wait logic.

Document parsing `model` is optional and defaults to `Model.PADDLE_OCR_VL_15`.

## Document Parsing

```python
from paddleocr import APIClient, DocParsingOptions, Model

client = APIClient()
result = client.parse_document(
    file_path="./report.pdf",
    options=DocParsingOptions(
        use_chart_recognition=True,
        use_formula_recognition=True,
    ),
)

for page in result.pages:
    print(page.markdown_text)
    print(page.markdown_images)
    print(page.output_images)
```

Supported document parsing models include `Model.PP_STRUCTURE_V3`,
`Model.PADDLE_OCR_VL`, and `Model.PADDLE_OCR_VL_15`.

## Submit, Status, And Wait

Convenience methods submit, wait, fetch, and parse in one call:

```python
ocr_result = client.ocr(file_url="https://example.com/a.pdf")
doc_result = client.parse_document(
    file_url="https://example.com/b.pdf",
)
```

Use explicit job control when you need non-blocking status checks or concurrent
waits:

```python
ocr_job = client.submit_ocr(file_url="https://example.com/a.pdf")
doc_job = client.submit_document_parsing(
    file_path="./b.pdf",
)

status = client.get_status(ocr_job.job_id)
print(status.state, status.progress)

ocr_result = client.wait_ocr_result(ocr_job)
doc_result = client.wait_document_parsing_result(doc_job)
```

`get_status(job_id)` performs one non-blocking status request. It does not poll
to a terminal state and does not download result payloads.

The async client exposes async versions of job operations: `ocr`,
`parse_document`, `submit_ocr`, `submit_document_parsing`, `get_status`,
`wait_ocr_result`, `wait_document_parsing_result`, and `close`.

```python
async with AsyncAPIClient() as client:
    job = await client.submit_ocr(file_url="https://example.com/a.pdf")
    result = await client.wait_ocr_result(job)
```

## Save Resources

Resource saving is currently provided by the sync `APIClient.save_resource`
method. `AsyncAPIClient` does not provide a `save_resource` coroutine method;
async users can use `APIClient.save_resource` after obtaining result resource
URLs, or implement custom async downloads if they need end-to-end async I/O.

`APIClient.save_resource(resource, destination, overwrite=False)` accepts either
a single resource URL or an `OCRResult` / `DocParsingResult`.

```python
saved_path = client.save_resource(
    result.pages[0].ocr_image_url,
    "./outputs/page-1.png",
    overwrite=False,
)
```

For result objects, `destination` must be an existing directory and the return
value is a `ResourceSaveSummary`:

```python
summary = client.save_resource(result, "./outputs", overwrite=False)
print(summary.saved_paths)
```

The helper never overwrites existing files unless `overwrite=True` is passed and
does not send PaddleOCR official API authorization headers to result resource URLs.

## CLI

The PaddleOCR CLI registers the official API client under the `api` subcommand:

```bash
paddleocr api \
  --model_type ocr \
  --file_url https://example.com/invoice.pdf \
  --request_timeout 300 \
  --poll_timeout 600 \
  --output result.json
```

For document parsing:

```bash
paddleocr api \
  --model_type document_parsing \
  --model PaddleOCR-VL-1.5 \
  --file_path ./report.pdf \
  --use_chart_recognition
```

## Timeouts

```python
client = APIClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` limits one HTTP request, including submit, status, and result
download calls. `poll_timeout` limits the total wait loop for `ocr`,
`parse_document`, `wait_ocr_result`, and
`wait_document_parsing_result`.

## Build And Test

Run the Python API SDK validation commands from the worktree root:

```bash
python -m pytest tests/test_api_client -q
python -c "import paddleocr; print(paddleocr.__version__)"
```

## Errors

All SDK errors inherit from `PaddleOCRAPIError`:

| Error | Meaning |
| --- | --- |
| `AuthError` | Missing token or HTTP 401/403 |
| `InvalidRequestError` | Invalid SDK input or HTTP 400 |
| `APIError` | Other non-2xx HTTP response |
| `NetworkError` | Transport failure before a response |
| `JobFailedError` | Remote job reached `failed` |
| `RequestTimeoutError` | One HTTP request exceeded `request_timeout` |
| `PollTimeoutError` | Wait loop exceeded `poll_timeout` |
| `FileNotFoundError` | Local file or destination parent is missing |
| `ResponseFormatError` | Successful API response is malformed |
| `ResultParseError` | Result JSONL cannot be parsed |

```python
from paddleocr import FileNotFoundError

try:
    client.ocr(file_path="./missing.pdf")
except FileNotFoundError as exc:
    print(exc.path)
    raise
```
