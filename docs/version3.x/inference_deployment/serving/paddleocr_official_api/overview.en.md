---
comments: true
---

# PaddleOCR official API Overview

The PaddleOCR official API SDKs are client libraries for the PaddleOCR official API. They submit local files or file URLs to hosted PaddleOCR services, poll asynchronous jobs, and parse typed results. They do not run local PaddleOCR inference or load local models.

The current clients include Python, TypeScript, Go, and the `paddleocr api` command in the PaddleOCR CLI.

## Authentication

First obtain an access token from the [AI Studio Access Token page](https://aistudio.baidu.com/account/accessToken).

All SDKs and the CLI read `PADDLEOCR_ACCESS_TOKEN` by default:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

You can also pass the token explicitly when constructing a client or invoking the CLI. Missing or invalid credentials are reported through typed authentication errors.

## Documentation Entry Points

- [Python SDK](python.en.md): best for projects already using the `paddleocr` Python package; provides sync `PaddleOCRClient` and async `AsyncPaddleOCRClient`.
- [TypeScript SDK](typescript.en.md): best for Node.js 18+ server-side projects.
- [Go SDK](go.en.md): best for statically typed services that need context cancellation and binary deployment.
- [CLI](cli.en.md): best for scripts, debugging, and quick no-code validation.

## Choose Models

The OCR model parameter is optional and defaults to PP-OCRv5. PP-OCRv5 is the only OCR model exposed by the current PaddleOCR official API release.

Document parsing model parameters are optional and default to PaddleOCR-VL-1.6 in every SDK. Supported models include PP-StructureV3, PaddleOCR-VL, PaddleOCR-VL-1.5, and PaddleOCR-VL-1.6.

## Results And Resources

Each SDK provides convenience calls, explicit submission, status queries, typed waits, and resource-saving methods. The CLI can output JSON and save resources referenced by the result object to a local directory. See the language-specific pages for errors, timeouts, and usage details.
