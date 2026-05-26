---
comments: true
---

# PaddleOCR official API TypeScript SDK

The TypeScript SDK targets Node.js 18+ and calls the PaddleOCR official API for OCR and document parsing. It uses hosted PaddleOCR services and does not run local PaddleOCR inference.

## Install And Authenticate

```bash
npm install paddleocr-sdk
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

The client reads `PADDLEOCR_ACCESS_TOKEN` by default and also accepts `token`:

```ts
import { PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient({
  token: process.env.PADDLEOCR_ACCESS_TOKEN,
});
```

## Quick Start

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  fileUrl: "https://example.com/invoice.pdf",
  model: Model.PPOCRv5,
});
console.log(result.jobId, result.pages.length);
```

Use `filePath` for a local file. Pass exactly one of `fileUrl` or `filePath`.

Document parsing example:

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.parseDocument({
  model: Model.PaddleOCRVL15,
  filePath: "./report.pdf",
  options: { useChartRecognition: true },
});
console.log(result.jobId, result.pages.length);
```

## Public API

The final TypeScript public methods are:

- `ocr(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `parseDocument(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `submitOcr(...)`: submit only an OCR job and return a job object.
- `submitDocumentParsing(...)`: submit only a document parsing job and return a job object.
- `getStatus(jobId)`: perform one non-blocking status request.
- `waitOcrResult(job)`: wait for an OCR job and parse its result.
- `waitDocumentParsingResult(job)`: wait for a document parsing job and parse its result.
- `saveResource(resource, destination, options)`: save one resource URL or resources referenced by a result object.

## Timeouts

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` limits one HTTP request, including submit, status, and resource downloads. `pollTimeout` limits the total wait time for `ocr`, `parseDocument`, `waitOcrResult`, and `waitDocumentParsingResult`. Public methods can also accept an `AbortSignal` for caller-driven cancellation.

## Model Extensibility

The OCR `model` is optional and defaults to `Model.PPOCRv5`. PP-OCRv5 is the only OCR model exposed by the current PaddleOCR official API release.

Document parsing `model` is optional and defaults to `Model.PaddleOCRVL15`. Supported document parsing models include `Model.PPStructureV3`, `Model.PaddleOCRVL`, and `Model.PaddleOCRVL15`. The SDK validates model categories through `isOCRModel` and `isDocumentParsingModel`, so future models can be added centrally.

## Errors And Resource Saving

All SDK errors inherit from `PaddleOCRAPIError`. Common typed errors include `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

`saveResource` can save one resource URL or all resources referenced by an `OCRResult` or `DocParsingResult`. It does not overwrite existing files by default and does not send PaddleOCR official API authorization headers to result resource URLs.

See the [api_sdk/typescript/README.md](../../../api_sdk/typescript/README.md) source-adjacent reference.
