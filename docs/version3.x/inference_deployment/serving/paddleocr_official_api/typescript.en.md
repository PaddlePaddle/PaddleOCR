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

## Choose Models

The `Model` enum values in the table are type-safe aliases for the official API model-name strings. They are serialized to the corresponding model name when the request is submitted. You can also pass the official API model-name string directly, for example `model: "PaddleOCR-VL-1.6"`.

| Task | Interfaces | Default model | Supported models | Option type |
| --- | --- | --- | --- | --- |
| OCR | `ocr`, `submitOcr`, `waitOcrResult` | `Model.PPOCRv5` | `Model.PPOCRv5` | `OCROptions` |
| Document parsing | `parseDocument`, `submitDocumentParsing`, `waitDocumentParsingResult` | `Model.PaddleOCRVL16` | `Model.PPStructureV3`, `Model.PaddleOCRVL`, `Model.PaddleOCRVL15`, `Model.PaddleOCRVL16` | Use `PPStructureV3Options` with `PPStructureV3`, and `PaddleOCRVLOptions` with PaddleOCR-VL models. |

Common mappings: `Model.PPOCRv5` maps to `PP-OCRv5`, `Model.PPStructureV3` maps to `PP-StructureV3`, `Model.PaddleOCRVL` maps to `PaddleOCR-VL`, `Model.PaddleOCRVL15` maps to `PaddleOCR-VL-1.5`, and `Model.PaddleOCRVL16` maps to `PaddleOCR-VL-1.6`.

## Configuration

### Client Configuration

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` limits one HTTP request, including submit, status, and resource downloads. `pollTimeout` limits the total wait time for `ocr`, `parseDocument`, `waitOcrResult`, and `waitDocumentParsingResult`. Public methods can also accept an `AbortSignal` for caller-driven cancellation.

Override the service base URL via the `PADDLEOCR_BASE_URL` environment variable or the `baseUrl` option:

```ts
const client = new PaddleOCRClient({
  baseUrl: "https://my-proxy.com/paddle",
});
```

Inject a custom `fetch` implementation for proxies or custom network layers:

```ts
const client = new PaddleOCRClient({
  fetch: myCustomFetch,
});
```

### Request Options

TypeScript SDK field names use camelCase, matching the official API directly. Omitted fields are not sent. See the interface source or Official API Reference for complete field definitions.

#### OCROptions (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `useDocOrientationClassify` | boolean | Document orientation classification |
| `useDocUnwarping` | boolean | Document unwarping |
| `visualize` | boolean | Return visualization images |

#### PPStructureV3Options (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `useTableRecognition` | boolean | Table recognition |
| `useFormulaRecognition` | boolean | Formula recognition |
| `useChartRecognition` | boolean | Chart recognition |
| `prettifyMarkdown` | boolean | Markdown prettification |

#### PaddleOCRVLOptions (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `useLayoutDetection` | boolean | Layout detection |
| `useChartRecognition` | boolean | Chart recognition |
| `temperature` | number | Sampling temperature |
| `prettifyMarkdown` | boolean | Markdown prettification |

## Error Handling

All SDK errors inherit from `PaddleOCRAPIError`. Common typed errors include `AuthError`, `InvalidRequestError`, `RateLimitError`, `ServiceUnavailableError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

## Official API Reference

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Dmh4onssk)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/7mfz6dgx9)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/Vmkz2nz1p)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/fml7mozw5)

## Quota and Error Codes

- [API Quota Rules and Error Code Description](https://ai.baidu.com/ai-doc/AISTUDIO/pmjcld5qm)
