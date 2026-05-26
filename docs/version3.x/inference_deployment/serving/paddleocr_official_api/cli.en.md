---
comments: true
---

# PaddleOCR official API CLI

`paddleocr api` is the PaddleOCR CLI subcommand for calling the PaddleOCR official API. It submits a file URL or local file to hosted services, waits for completion, and outputs JSON. It does not run local inference.

## Authentication

The CLI reads `PADDLEOCR_ACCESS_TOKEN` by default:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

You can also pass a token explicitly with `--token`.

## Basic Usage

```bash
paddleocr api \
  --model_type ocr \
  --file_url https://example.com/invoice.pdf
```

`--model_type` is required and accepts `ocr` or `document_parsing`. Pass exactly one of `--file_url` or `--file_path`.

## Common Options

- `--model_type`: task type, either `ocr` or `document_parsing`.
- `--model`: model name. OCR defaults to PP-OCRv5; document parsing defaults to PaddleOCR-VL-1.6 when omitted. Models are validated through the PaddleOCR official API SDK model classification helpers.
- `--file_url`: file URL to process.
- `--file_path`: local file path to upload and process.
- `--request_timeout`: timeout in seconds for one HTTP request.
- `--poll_timeout`: total timeout in seconds while waiting for the remote job to complete.
- `--output`: JSON output file path; omitted means print to stdout.
- `--page_ranges`: page ranges such as `2,4-6`.
- `--use_doc_orientation_classify`, `--use_doc_unwarping`, `--use_textline_orientation`: optional OCR-related capabilities.
- `--use_chart_recognition`: optional document parsing capability.

## OCR Example

```bash
paddleocr api \
  --model_type ocr \
  --model PP-OCRv5 \
  --file_path ./invoice.pdf \
  --request_timeout 300 \
  --poll_timeout 600 \
  --output ocr-result.json
```

## Document Parsing Example

```bash
paddleocr api \
  --model_type document_parsing \
  --file_url https://example.com/report.pdf \
  --use_chart_recognition \
  --output doc-result.json
```

## Output Behavior

On success, the command emits formatted JSON. OCR output includes `jobId` plus each page's `prunedResult` and `ocrImageUrl`; document parsing output includes `jobId` plus each page's `markdownText`, `markdownImages`, and `outputImages`. With `--output`, the CLI writes that file and prints the saved path; otherwise it prints JSON to stdout.

Errors are printed to stderr and return a non-zero exit code. Common causes include a missing `PADDLEOCR_ACCESS_TOKEN`, a model that does not match `--model_type`, request timeout, poll timeout, failed remote job, or malformed response.
