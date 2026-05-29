---
comments: true
---

# PaddleOCR official API CLI

`paddleocr api` is the PaddleOCR CLI subcommand for calling the PaddleOCR official API. It submits a file URL or local file to hosted services, waits for completion, and outputs the parsing result. It does not run local inference.

## Install And Authenticate

First install the Python package by following [Install paddleocr](../../../installation.en.md#11-install-paddleocr). After installing the core `paddleocr` package, this feature is available out of the box — no additional dependency groups are required.

First obtain an access token from the [AI Studio Access Token page](https://aistudio.baidu.com/account/accessToken).

The CLI reads `PADDLEOCR_ACCESS_TOKEN` by default:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

You can also pass a token explicitly with `--token`.

## Basic Usage

```bash
paddleocr api \
  --model_type ocr \
  --file_url https://example.com/invoice.pdf
```

`--model_type` is required and accepts `ocr` or `doc_parsing`. Pass exactly one of `--file_url` or `--file_path`.

## Common Options

- `--model_type`: task type, either `ocr` or `doc_parsing`.
- `--model`: model name. OCR defaults to PP-OCRv5; document parsing defaults to PaddleOCR-VL-1.6 when omitted.
- `--file_url`: file URL to process.
- `--file_path`: local file path to upload and process.
- `--base_url`: base URL of the PaddleOCR API service; defaults to the official service URL (can also be set via `PADDLEOCR_BASE_URL`).
- `--request_timeout`: timeout in seconds for one HTTP request.
- `--poll_timeout`: total timeout in seconds while waiting for the remote job to complete.
- `--output`: JSON output file path; omitted means print to stdout.
- `--save_resources`: directory for saving resources referenced by the result object.
- `--overwrite_resources`: overwrite existing files when saving resources.
- `--page_ranges`: page ranges such as `2,4-6`.
- `--use_doc_orientation_classify` / `--no-use_doc_orientation_classify`: enable or disable document orientation classification.
- `--use_doc_unwarping` / `--no-use_doc_unwarping`: enable or disable document unwarping.
- `--use_textline_orientation` / `--no-use_textline_orientation`: enable or disable textline orientation detection.
- `--text_det_limit_side_len`: image side length limit for text detection.
- `--text_det_limit_type`: side length limit type, `min` or `max`.
- `--text_rec_score_thresh`: score threshold for text recognition results.
- `--use_layout_detection` / `--no-use_layout_detection`: enable or disable layout detection.
- `--use_seal_recognition` / `--no-use_seal_recognition`: enable or disable seal recognition.
- `--use_table_recognition` / `--no-use_table_recognition`: enable or disable table recognition.
- `--use_formula_recognition` / `--no-use_formula_recognition`: enable or disable formula recognition.
- `--use_chart_recognition` / `--no-use_chart_recognition`: enable or disable chart recognition.
- `--visualize` / `--no-visualize`: enable or disable result visualization images.
- `--prettify_markdown` / `--no-prettify_markdown`: enable or disable markdown prettification.

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
  --model_type doc_parsing \
  --file_url https://example.com/report.pdf \
  --use_chart_recognition \
  --save_resources ./doc-assets \
  --output doc-result.json
```

## Choose Models

| Task | `--model_type` | Default model | Supported models |
| --- | --- | --- | --- |
| OCR | `ocr` | `PP-OCRv5` | `PP-OCRv5` |
| Document parsing | `doc_parsing` | `PaddleOCR-VL-1.6` | `PP-StructureV3`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`, `PaddleOCR-VL-1.6` |

## Output Behavior

On success, the command emits formatted JSON. OCR output includes `jobId` plus each page's `prunedResult` and `ocrImageUrl`; document parsing output includes `jobId` plus each page's `markdownText`, `markdownImages`, and `outputImages`. With `--output`, the CLI writes that file and prints the saved path; otherwise it prints JSON to stdout. With `--save_resources`, the CLI saves resources referenced by the result object into the target directory.

Errors are printed to stderr and return a non-zero exit code. Common causes include a missing `PADDLEOCR_ACCESS_TOKEN`, a model that does not match `--model_type`, request timeout, poll timeout, failed remote job, or malformed response.

## Official API Reference

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Dmh4onssk)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/7mfz6dgx9)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/Vmkz2nz1p)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/fml7mozw5)

## Quota and Error Codes

- [API Quota Rules and Error Code Description](https://ai.baidu.com/ai-doc/AISTUDIO/pmjcld5qm)
