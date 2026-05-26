# PaddleOCR official API SDKs

The PaddleOCR official API SDKs are client libraries for the PaddleOCR official API. They
submit files or file URLs to hosted PaddleOCR services, poll asynchronous jobs,
fetch result resources, and return typed result objects. They do not run local
PaddleOCR inference, load local models, or provide offline OCR execution.

The official user documentation is published through the mkdocs site under
[`docs/version3.x/api_sdk/`](../docs/version3.x/api_sdk/):

- [Overview](../docs/version3.x/api_sdk/overview.md)
- [Python SDK](../docs/version3.x/api_sdk/python.md)
- [TypeScript SDK](../docs/version3.x/api_sdk/typescript.md)
- [Go SDK](../docs/version3.x/api_sdk/go.md)
- [CLI](../docs/version3.x/api_sdk/cli.md)

The files under `api_sdk/` are source-adjacent references for SDK maintainers and
release work.

## Languages

| Language | Package location | Status |
| --- | --- | --- |
| Python | `paddleocr._api_client` | Release candidate |
| TypeScript | `api_sdk/typescript` | Release candidate |
| Go | `api_sdk/go` | Release candidate |

## Install And Authenticate

All SDKs read `PADDLEOCR_ACCESS_TOKEN` by default. You may also pass the token
explicitly when constructing a client.

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

Python is bundled with the PaddleOCR package in this worktree:

```bash
python -m pip install -e .
```

TypeScript requires Node.js 18 or newer:

```bash
cd api_sdk/typescript
npm install
npm run build
```

Go can be used from the module path after the SDK is published:

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
```

## Quick Start

Python:

```python
from paddleocr import APIClient, Model

client = APIClient()
result = client.ocr(
    file_url="https://example.com/invoice.pdf",
    model=Model.PP_OCRV5,
)
print(result.job_id, len(result.pages))
```

TypeScript:

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  fileUrl: "https://example.com/invoice.pdf",
  model: Model.PPOCRv5,
});
console.log(result.jobId, result.pages.length);
```

Go:

```go
client, err := paddleocr.NewClient()
if err != nil {
	return err
}
result, err := client.OCR(ctx, &paddleocr.OCRRequest{
	Model:   paddleocr.PPOCRv5,
	FileURL: "https://example.com/invoice.pdf",
})
```

OCR model parameters are optional and default to PP-OCRv5 in every SDK. PP-OCRv5
is the only OCR model supported by the current PaddleOCR official API release;
each SDK validates OCR and document parsing models through centralized
classification helpers so future OCR models can be added without changing
submit/wait logic.

## Cross-Language API Names

| Operation | Python | TypeScript | Go |
| --- | --- | --- | --- |
| Create client | `APIClient(...)`, `AsyncAPIClient(...)` | `new PaddleOCRClient(...)` | `NewClient(...)` |
| Token option | `token`, `PADDLEOCR_ACCESS_TOKEN` | `token`, `PADDLEOCR_ACCESS_TOKEN` | `WithToken(...)`, `PADDLEOCR_ACCESS_TOKEN` |
| Base URL option | `base_url` | `baseUrl` | `WithBaseURL(...)` |
| Request timeout | `request_timeout` | `requestTimeout` | `WithRequestTimeout(...)` |
| Poll timeout | `poll_timeout` | `pollTimeout` | `WithPollTimeout(...)` |
| OCR convenience call | `ocr(...)` | `ocr(...)` | `OCR(...)` |
| Document parsing convenience call | `parse_document(...)` | `parseDocument(...)` | `ParseDocument(...)` |
| Submit OCR | `submit_ocr(...)` | `submitOcr(...)` | `SubmitOCR(...)` |
| Submit document parsing | `submit_document_parsing(...)` | `submitDocumentParsing(...)` | `SubmitDocumentParsing(...)` |
| Non-blocking status | `get_status(job_id)` | `getStatus(jobId)` | `GetStatus(ctx, jobID)` |
| Wait OCR result | `wait_ocr_result(job)` | `waitOcrResult(job)` | `WaitOCRResult(ctx, job)` / `Operation.WaitOCR(ctx)` |
| Wait document parsing result | `wait_document_parsing_result(job)` | `waitDocumentParsingResult(job)` | `WaitDocumentParsingResult(ctx, job)` / `Operation.WaitDocumentParsing(ctx)` |
| Save/download resource | `save_resource(...)` | `saveResource(...)` | `SaveResource(...)`, `SaveOCRResultResources(...)`, `SaveDocumentParsingResultResources(...)` |

`get_status`, `getStatus`, and `GetStatus` are non-blocking status checks only.
Use the typed wait methods when you want the SDK to poll to completion and parse
the result payload.

## Feature Matrix

| Release bar | Python | TypeScript | Go |
| --- | --- | --- | --- |
| Reads `PADDLEOCR_ACCESS_TOKEN` and rejects missing auth | Done | Done | Done |
| URL OCR and local file OCR | Done | Done | Done |
| URL/file document parsing | Done | Done | Done |
| Typed async submit/status/wait APIs | Done | Done | Done |
| Separate request and poll timeouts | Done | Done | Done |
| Typed errors for auth, validation, HTTP, network, timeout, job failure, response format, and result parsing | Done | Done | Done |
| Result URL downloads omit API authorization | Done | Done | Done |
| Resource URL saving helper | Done | Done | Done |
| Result-object bulk resource saving | Done | Done | Done |
| Contract-approved examples and docs | Done | Done | Done |

The TypeScript `saveResource` helper accepts either one resource URL or an
`OCRResult` / `DocParsingResult` for bulk resource saving into an existing
directory. Go exposes typed bulk helpers:
`SaveOCRResultResources` and `SaveDocumentParsingResultResources`.

## Validation Commands

Run these from the worktree root unless a subdirectory is shown:

```bash
python -m pytest tests/test_api_client -q
python -c "import paddleocr; print(paddleocr.__version__)"

cd api_sdk/typescript
npm run lint
npm run build
npm test
npm audit --audit-level=moderate

cd ../go
go test ./...
go vet ./...

cd ../..
git diff --check
git status --short --ignored=matching -- api_sdk/typescript
```

See `RELEASE_CHECKLIST.md` for the full release readiness gate and
`BENCHMARK_EVALUATION.md` for the MinerU-Ecosystem comparison.
