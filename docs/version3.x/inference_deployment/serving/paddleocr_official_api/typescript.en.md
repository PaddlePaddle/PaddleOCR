---
comments: true
---

# PaddleOCR official API TypeScript SDK

The TypeScript SDK targets Node.js 18+ and calls the PaddleOCR official API for OCR and document parsing. It uses hosted PaddleOCR services and does not run local PaddleOCR inference.

## Install And Authenticate

First obtain an access token from the [AI Studio Access Token page](https://aistudio.baidu.com/account/accessToken).

```bash
npm install @paddleocr/api-sdk
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

The client reads `PADDLEOCR_ACCESS_TOKEN` by default and also accepts `token`:

```ts
import { PaddleOCRClient } from "@paddleocr/api-sdk";

const client = new PaddleOCRClient({
  token: process.env.PADDLEOCR_ACCESS_TOKEN,
});
```

## Quick Start

```ts
import { Model, PaddleOCRClient } from "@paddleocr/api-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  fileUrl: "https://example.com/invoice.pdf",
  model: Model.PPOCRv5,
});
console.log(result.jobId, result.pages.length);
```

Use `filePath` for a local file. Pass exactly one of `fileUrl` or `filePath`.

## Public API

Common TypeScript public methods include:

- `ocr(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `parseDocument(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `submitOcr(...)`: submit only an OCR job and return a job object.
- `submitDocumentParsing(...)`: submit only a document parsing job and return a job object.
- `getStatus(jobId)`: perform one non-blocking status request.
- `waitOcrResult(job)`: wait for an OCR job and parse its result.
- `waitDocumentParsingResult(job)`: wait for a document parsing job and parse its result.
- `saveResource(resourceUrl, destination, options)`: save one resource URL.
- `saveOcrResultResources(result, destination, options)`: save resources referenced by an OCR result object.
- `saveDocumentParsingResultResources(result, destination, options)`: save resources referenced by a document parsing result object.

## Timeouts

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` limits one HTTP request, including submit, status, and resource downloads. `pollTimeout` limits the total wait time for `ocr`, `parseDocument`, `waitOcrResult`, and `waitDocumentParsingResult`. Public methods can also accept an `AbortSignal` for caller-driven cancellation.

## Choose Models

| Task | Interfaces | Default model | Supported models | Option type |
| --- | --- | --- | --- | --- |
| OCR | `ocr`, `submitOcr`, `waitOcrResult` | `Model.PPOCRv5` | `Model.PPOCRv5` | `OCROptions` |
| Document parsing | `parseDocument`, `submitDocumentParsing`, `waitDocumentParsingResult` | `Model.PaddleOCRVL16` | `Model.PPStructureV3`, `Model.PaddleOCRVL`, `Model.PaddleOCRVL15`, `Model.PaddleOCRVL16` | Use `PPStructureV3Options` with `PPStructureV3`, and `PaddleOCRVLOptions` with PaddleOCR-VL models. |

## Errors And Resource Saving

All SDK errors inherit from `PaddleOCRAPIError`. Common typed errors include `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

Use `saveResource` for one resource URL. To save all resources referenced by a result object, use `saveOcrResultResources` or `saveDocumentParsingResultResources`.
