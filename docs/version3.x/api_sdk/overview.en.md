---
comments: true
---

# PaddleOCR official API Overview

The PaddleOCR official API SDKs are client libraries for the PaddleOCR official API. They submit local files or file URLs to hosted PaddleOCR services, poll asynchronous jobs, and parse typed results. They do not run local PaddleOCR inference or load local models.

The current clients include Python, TypeScript, Go, and the `paddleocr api` command in the PaddleOCR CLI. Source-adjacent references live in the [Python SDK reference](../../../api_sdk/PYTHON.md), [TypeScript SDK reference](../../../api_sdk/typescript/README.md), and [Go SDK reference](../../../api_sdk/go/README.md). The user docs entry points are:

- [Python SDK](./python.en.md)
- [TypeScript SDK](./typescript.en.md)
- [Go SDK](./go.en.md)
- [CLI](./cli.en.md)

## Authentication

All SDKs and the CLI read `PADDLEOCR_ACCESS_TOKEN` by default:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

You can also pass the token explicitly when constructing a client or invoking the CLI. Missing or invalid credentials are reported through typed authentication errors.

## Choose A Client

- Python: best for projects already using the `paddleocr` Python package; provides sync `APIClient` and async `AsyncAPIClient`.
- TypeScript: best for Node.js 18+ server-side projects.
- Go: best for statically typed services that need context cancellation and binary deployment.
- CLI: best for scripts, debugging, and quick no-code validation.

## Models And Tasks

The OCR model parameter is optional and defaults to PP-OCRv5. PP-OCRv5 is the only OCR model exposed by the current PaddleOCR official API release. Each SDK validates OCR and document parsing models through centralized classification helpers, so future OCR models can be added centrally without changing submit, polling, or resource-saving flows.

Document parsing model parameters are optional and default to `PaddleOCR-VL-1.5`. Supported models include `PP-StructureV3`, `PaddleOCR-VL`, and `PaddleOCR-VL-1.5`. When `--model` or `model` is omitted, each SDK and the CLI default to `PaddleOCR-VL-1.5`.

## Results And Resources

Convenience methods submit a job, wait for completion, download result JSONL, and parse typed result objects. Explicit submit methods are useful when you need non-blocking status checks, concurrent waits, or custom scheduling.

Output images, Markdown assets, and other result resources can be saved with helpers such as `save_resource`, `saveResource`, and `SaveResource`. Resource downloads do not send PaddleOCR official API authorization headers to result URLs and do not overwrite existing files by default.

## Errors

At the user-documentation level, SDKs expose typed errors for authentication, validation, non-2xx HTTP responses, network failures, request timeouts, poll timeouts, failed remote jobs, malformed responses, and result parsing failures. The CLI prints errors to stderr and exits with a non-zero status.
