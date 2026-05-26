# PaddleOCR TypeScript SDK

TypeScript client for the PaddleOCR official API. It requires Node.js 18 or newer
and uses hosted PaddleOCR services; it does not perform local OCR inference.

This file is a source-adjacent reference for TypeScript SDK maintenance. The
official user docs live in `docs/version3.x/api_sdk/typescript.md` and
`docs/version3.x/api_sdk/typescript.en.md`.

## Install

```bash
npm install paddleocr-sdk
```

For local development in this worktree:

```bash
npm install
npm run build
```

## Authentication

Set `PADDLEOCR_ACCESS_TOKEN` or pass `token` to the client:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

```ts
import { PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient({
  token: process.env.PADDLEOCR_ACCESS_TOKEN,
});
```

The constructor throws `AuthError` if no token is supplied.

## OCR From URL

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  model: Model.PPOCRv5,
  fileUrl: "https://example.com/invoice.pdf",
});

for (const page of result.pages) {
  console.log(page.prunedResult);
  console.log(page.ocrImageUrl);
}
```

## OCR From File

```ts
const result = await client.ocr({
  filePath: "./invoice.pdf",
  pageRanges: "1-3",
  options: {
    useDocOrientationClassify: true,
    useTextlineOrientation: true,
  },
});
```

Pass exactly one of `fileUrl` or `filePath`.

`model` is optional for OCR and defaults to `Model.PPOCRv5`. PP-OCRv5 is the
only OCR model supported by the current PaddleOCR official API release; model
validation is centralized in `isOCRModel` and `isDocumentParsingModel` so future
OCR models can be added without changing submit/wait logic.

Document parsing `model` is optional and defaults to `Model.PaddleOCRVL15`.

## Document Parsing

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.parseDocument({
  filePath: "./report.pdf",
  options: {
    useChartRecognition: true,
    useFormulaRecognition: true,
  },
});

for (const page of result.pages) {
  console.log(page.markdownText);
  console.log(page.markdownImages);
  console.log(page.outputImages);
}
```

Supported document parsing models are `Model.PPStructureV3`,
`Model.PaddleOCRVL`, and `Model.PaddleOCRVL15`.

## Submit, Status, And Wait

Use the convenience methods when you want one call to submit, poll, fetch, and
parse:

```ts
const ocrResult = await client.ocr({ fileUrl: "https://example.com/a.pdf" });
const docResult = await client.parseDocument({
  model: Model.PaddleOCRVL15,
  fileUrl: "https://example.com/b.pdf",
});
```

Use explicit job control when you need status checks or concurrent waits:

```ts
const ocrJob = await client.submitOcr({
  fileUrl: "https://example.com/a.pdf",
});
const docJob = await client.submitDocumentParsing({
  model: Model.PaddleOCRVL15,
  filePath: "./b.pdf",
});

const status = await client.getStatus(ocrJob.jobId);
console.log(status.state, status.progress);

const [ocrResult, docResult] = await Promise.all([
  client.waitOcrResult(ocrJob),
  client.waitDocumentParsingResult(docJob),
]);
```

`getStatus(jobId)` performs one non-blocking status request. It does not wait
for completion or download result payloads.

## Save Resources

`saveResource(resourceUrl, destination, options)` downloads a single result
resource URL and returns a structured summary:

```ts
const saved = await client.saveResource(
  result.pages[0].ocrImageUrl!,
  "./outputs",
  { overwrite: false, filename: "page-1.png" },
);
console.log(saved.savedPaths);
```

If `destination` is an existing directory, the helper derives the filename from
the URL or `options.filename`. Existing files are not overwritten unless
`overwrite: true` is set. Resource downloads do not send the PaddleOCR official API
authorization header to result URLs.

`saveResource(result, destination, options)` also accepts an `OCRResult` or
`DocParsingResult` and saves all referenced resources into an existing directory:

```ts
const ocrSaved = await client.saveResource(ocrResult, "./ocr-images");
const docSaved = await client.saveResource(docResult, "./doc-assets", {
  overwrite: true,
});
console.log(ocrSaved.savedPaths, docSaved.savedPaths);
```

For OCR results, each page `ocrImageUrl` is saved as `ocr-page-{pageNumber}` with
the URL path extension when available. For document parsing results,
`markdownImages` and `outputImages` are saved with their service map keys as
filenames when safe; spaces and normal punctuation are preserved, and URL
basenames are only used by the single-URL mode. Map keys are saved in sorted
order for deterministic output. Keys that are empty, absolute, contain a `..`
path traversal segment, or contain path separators are rejected with
`InvalidRequestError`. Bulk saving requires
`destination` to already be a directory and does not overwrite existing files
unless `overwrite: true` is set.

## Timeouts

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` limits one HTTP request, including submit, status, and result
download calls. `pollTimeout` limits the total wait loop for
`ocr`, `parseDocument`, `waitOcrResult`, and `waitDocumentParsingResult`.

You may also pass an `AbortSignal` to public methods:

```ts
const controller = new AbortController();
const result = await client.ocr(
  { fileUrl: "https://example.com/a.pdf" },
  { signal: controller.signal },
);
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
| `RequestTimeoutError` | One HTTP request exceeded `requestTimeout` |
| `PollTimeoutError` | Wait loop exceeded `pollTimeout` |
| `FileNotFoundError` | Local file or destination parent is missing |
| `ResponseFormatError` | Successful API response is malformed |
| `ResultParseError` | Result JSONL cannot be parsed |

```ts
try {
  await client.ocr({ filePath: "./missing.pdf" });
} catch (error) {
  if (error instanceof FileNotFoundError) {
    console.error(error.path);
  }
  throw error;
}
```

## Build And Test

```bash
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
```
