# PaddleOCR official API SDK Contract

This contract defines the first public release surface for the PaddleOCR official API SDKs. Because the current branch APIs are unpublished, implementation teams must rename, reshape, or remove existing draft APIs as needed to match this contract; no backward compatibility is required for unpublished branch APIs.

The SDKs are official API wrappers for PaddleOCR services. They do not run local PaddleOCR inference, load local PaddleOCR models, or provide offline OCR execution.

## Client Options

All SDKs must support the same client behavior with idiomatic language names.

| Option | Environment | Python | TypeScript | Go | Required behavior |
| --- | --- | --- | --- | --- | --- |
| API token / `PADDLEOCR_ACCESS_TOKEN` | `PADDLEOCR_ACCESS_TOKEN` | `token` | `token` | `Token` | Authenticates PaddleOCR official API requests. A missing token may be read from `PADDLEOCR_ACCESS_TOKEN`; if neither is present, authenticated clients must fail at client construction with an auth error unless an explicit no-auth mode is added in a future contract. |
| Base URL | None | `base_url` | `baseUrl` | `BaseURL` | Overrides the PaddleOCR official API endpoint. Defaults to the production endpoint documented by the SDK release. SDKs must normalize trailing slashes consistently so path joining does not produce double slashes or omit separators. |
| Request timeout | None | `request_timeout` | `requestTimeout` | `RequestTimeout` | Maximum duration for one HTTP request, including submit, status, and result download requests. This is separate from poll timeout. |
| Poll timeout | None | `poll_timeout` | `pollTimeout` | `PollTimeout` | Maximum total duration spent waiting for an asynchronous job to reach a terminal state. This is separate from request timeout. |

## Operations

Each SDK must expose the following operations with names that are idiomatic but semantically identical.

| Operation | Python | TypeScript | Go | Behavior |
| --- | --- | --- | --- | --- |
| OCR convenience call | `ocr(...)` | `ocr(...)` | `OCR(...)` | Submits an OCR job, waits for completion, downloads/parses the OCR result, and returns `OCRResult`. |
| Document parsing convenience call | `parse_document(...)` | `parseDocument(...)` | `ParseDocument(...)` | Submits a document parsing job, waits for completion, downloads/parses the document parsing result, and returns `DocParsingResult`. |
| Submit OCR | `submit_ocr(...)` | `submitOcr(...)` | `SubmitOCR(...)` | Starts an OCR job and returns `Job` without blocking for completion. |
| Submit document parsing | `submit_document_parsing(...)` | `submitDocumentParsing(...)` | `SubmitDocumentParsing(...)` | Starts a document parsing job and returns `Job` without blocking for completion. |
| Get status | `get_status(job_id)` | `getStatus(jobId)` | `GetStatus(ctx, jobID)` | Performs a non-blocking status request only and returns `JobStatus`. |
| Wait OCR result | `wait_ocr_result(job)` | `waitOcrResult(job)` | `WaitOCRResult(ctx, job)` | Polls an OCR job until completion and returns `OCRResult`. |
| Wait document parsing result | `wait_document_parsing_result(job)` | `waitDocumentParsingResult(job)` | `WaitDocumentParsingResult(ctx, job)` | Polls a document parsing job until completion and returns `DocParsingResult`. |
| Resource saving/downloading helper | `save_resource(...)` | `saveResource(...)` | `SaveResource(...)` | Downloads or saves result resources using the SDK's result URL handling rules and overwrite behavior. |

## Data Models

`Job` represents an accepted asynchronous hosted task. It must carry `jobId` plus model and task information sufficient for the SDK to know whether the job is OCR or document parsing without inspecting result payload fields.

`JobStatus` represents the current state of a job. It must carry `jobId`, `state`, `progress`, and an error message field for terminal failures. `progress` should preserve the service value when supplied and should be nullable or optional when the service omits it.

The baseline canonical job states are `pending`, `running`, `done`, and `failed`. `done` and `failed` are terminal states. Unknown states are `ResponseFormatError` unless the contract is deliberately extended later. If the service adds states such as `canceled` or `expired`, this contract must be updated before SDKs expose them as public states.

`OCRResult` and `DocParsingResult` must be typed per task. They may contain language-specific convenience structures, but the public shape must preserve the service result data needed by users to consume OCR and document parsing outputs without parsing raw HTTP responses themselves.

`OCRResult` must expose at least `jobId` and `pages`. Each OCR page must expose the pruned result or raw OCR payload as `prunedResult`, `raw`, or an equivalent documented language-idiomatic field, plus an optional OCR image URL when the service returns one.

`DocParsingResult` must expose at least `jobId` and `pages`. Each document parsing page must expose `markdownText` and resource maps for markdown/output images when the service returns them.

Result models may include a raw payload escape hatch for advanced users, but a raw-only public result must not be the only structured API.

Job states must be typed or enumerated where the language supports it. Python should expose a typed string literal or enum, TypeScript should expose a string union or enum, and Go should expose a named string type with constants.

## Model Classification

OCR APIs must accept a typed model parameter and default to PP-OCRv5:

- Python: `ocr(..., model=Model.PP_OCRV5)` and `submit_ocr(..., model=Model.PP_OCRV5)`.
- TypeScript: `OCRRequest.model?: Model`, defaulting to `Model.PPOCRv5`.
- Go: `OCRRequest.Model Model`, defaulting to `PPOCRv5` when zero-valued.

Document parsing APIs must accept a typed model parameter and default to PaddleOCR-VL-1.5:

- Python: `parse_document(..., model=Model.PADDLE_OCR_VL_15)` and `submit_document_parsing(..., model=Model.PADDLE_OCR_VL_15)`.
- TypeScript: `DocParsingRequest.model?: Model`, defaulting to `Model.PaddleOCRVL15`.
- Go: `DocParsingRequest.Model Model`, defaulting to `PaddleOCRVL15` when zero-valued.

PP-OCRv5 is the only OCR model supported by this release, but SDK submit/wait
logic must be model-extensible. Each SDK must centralize model classification in
helpers such as `is_ocr_model` / `is_document_parsing_model`,
`isOCRModel` / `isDocumentParsingModel`, and `IsOCRModel` /
`IsDocumentParsingModel`. OCR wait methods must validate jobs through the OCR
classification helper rather than direct PP-OCRv5 equality checks. Document
parsing submit/wait validation must reject OCR models through the same
classification layer.

## Resource Saving

The resource saving/downloading helper must accept a result object or resource URL plus a destination path, depending on language idiom. When a result object contains multiple downloadable resources, the helper may save all eligible resources or require a documented selector.

The helper must return the saved file path or paths, or a typed save summary containing the saved destinations and skipped resources. Return values must be structured enough for callers to know what was written.

The helper must not silently overwrite existing files. Overwrite is allowed only when the caller explicitly passes an overwrite option.

Network and download failures must map to the SDK's Network or API/HTTP errors according to the error precedence rules. Filesystem failures must surface as language-idiomatic filesystem errors or documented SDK errors.

## Naming Rules

`get_status`, `getStatus`, and `GetStatus` are reserved for non-blocking status APIs only. They must not wait for terminal completion and must not download or parse result payloads.

Status APIs must not be named `get_result`, `getResult`, or `GetResult`. Result-returning APIs must be explicit about waiting and task kind, such as `wait_ocr_result`, `waitOcrResult`, and `WaitOCRResult`.

SDKs must not infer result kind from JSONL field presence. Result parsing must be selected from explicit task/model information carried by `Job`, `JobStatus`, the wait method, or another documented typed discriminator.

Request timeout and poll timeout are separate concepts. A request timeout applies to a single HTTP operation; a poll timeout applies to the total wait loop for asynchronous completion.

SDKs must not preserve old draft aliases for unpublished branch compatibility. In particular, do not keep `get_result`, `getResult`, `GetResult`, or a single timeout option as compatibility aliases.

## Error Taxonomy

All SDKs must expose typed errors or documented error classes/categories for the same failure modes:

| Error category | Meaning |
| --- | --- |
| Auth | Missing, invalid, or rejected credentials, including absent `token` and `PADDLEOCR_ACCESS_TOKEN`. Public error name should be `AuthError` or the language-idiomatic equivalent. |
| Invalid request | User-provided inputs fail SDK-side validation or are rejected as invalid API parameters. Public error name should be `InvalidRequestError` or the language-idiomatic equivalent. |
| API/HTTP | The PaddleOCR official API returns a non-success HTTP status or documented API error response. |
| Network | DNS, connection, TLS, socket, or other transport failures before a valid HTTP response is received. |
| Job failed | A submitted job reaches a terminal failed state reported by the service. |
| Request timeout | A single HTTP request exceeds the configured request timeout. |
| Poll timeout | A wait operation exceeds the configured poll timeout before the job reaches a terminal state. |
| File not found | A local input path or destination parent path required by the SDK does not exist. Public error name should be `FileNotFoundError` or the language-idiomatic equivalent. |
| Response format | A transport-successful response is missing required fields, has unknown state values, or otherwise violates the documented API response schema. Public error name should be `ResponseFormatError` or the language-idiomatic equivalent. |
| Result parse | Result payload parsing fails after a result resource is fetched, including malformed JSONL. Public error name should be `ResultParseError` or the language-idiomatic equivalent. |

Error precedence must be consistent across languages:

- Missing local input files and missing destination parent directories are `FileNotFoundError` or the language-idiomatic equivalent, and take precedence over generic SDK-side `InvalidRequestError` validation.
- SDK-side semantic validation errors before network calls are `InvalidRequestError`, including both file URL and file path provided, neither file URL nor file path provided, unsupported model, wait-method task mismatch, and invalid page range format when validated locally.
- Missing token for authenticated clients fails at client construction as `AuthError`, unless an explicit no-auth mode is added in a future contract.
- HTTP 401 and 403 are `AuthError`.
- HTTP 400 from the server is `InvalidRequestError` with the server message preserved when available.
- Other non-2xx HTTP responses are API errors.
- HTTP 2xx responses with malformed bodies are `ResponseFormatError`.
- JSONL or result payload parse failures are `ResultParseError`.
- Passing an OCR job to a document parsing wait method, or a document parsing job to an OCR wait method, is `InvalidRequestError`.

## Cross-Language Behavior Rules

HTTP 2xx means transport success only. SDKs must still validate the response body against the expected API schema before returning public data models.

Malformed successful responses are `ResponseFormatError`. This includes missing `jobId`, missing status state, a done job without a result URL, or any other required field violation.

Malformed JSONL is a `ResultParseError`. JSONL parsing failures must not be reported as API/HTTP errors once the result resource has been fetched successfully.

JSONL/result URL fetches must not send Authorization headers to presigned URLs, object storage URLs, or other result download URLs outside the PaddleOCR official API origin. Authorization is only for PaddleOCR official API requests.

A done job without a result URL is a `ResponseFormatError`.

Unknown job states are `ResponseFormatError` unless the service documentation explicitly defines an extension mechanism and the SDK documents how unknown states are represented.

## Release Acceptance Checklist

Implementation teams must complete this checklist before the first public SDK release:

- Public client option names match this contract in Python, TypeScript, and Go.
- Missing `PADDLEOCR_ACCESS_TOKEN` fallback fails at authenticated client construction unless an explicit no-auth mode is added in a future contract.
- Base URL normalization handles trailing slashes consistently.
- Request timeout and poll timeout are configured and tested as separate behaviors.
- Operation names and blocking behavior match the operations table.
- `get_status`, `getStatus`, and `GetStatus` perform non-blocking status checks only.
- No status API uses `get_result`, `getResult`, or `GetResult` naming, and no old draft aliases or single timeout compatibility option are preserved.
- `Job`, `JobStatus`, `OCRResult`, and `DocParsingResult` are typed and documented in each language.
- `OCRResult` and `DocParsingResult` expose the minimum structured fields required by this contract, not only raw payloads.
- Job states are typed or enumerated where the language supports it, with `pending`, `running`, `done`, and `failed` baseline semantics.
- OCR APIs expose a typed model parameter, default to PP-OCRv5, and validate via centralized OCR/document parsing model classification helpers.
- `done` and `failed` are terminal, and unknown states produce `ResponseFormatError`.
- Result parsing is selected from explicit task/model information, not JSONL field presence.
- Wait methods reject mismatched task kinds with `InvalidRequestError`.
- Error precedence is implemented and tested for file-not-found precedence, SDK semantic validation, 401/403, 400, other non-2xx, malformed 2xx, and result parse failures.
- Malformed 2xx API responses are reported as `ResponseFormatError`.
- Malformed JSONL payloads are reported as `ResultParseError`.
- Result URL downloads omit Authorization for presigned/object storage URLs.
- Resource saving accepts documented result object or URL inputs, returns saved paths or a typed summary, requires explicit overwrite, and maps download/filesystem failures as documented.
- Done jobs without result URLs and unknown job states are covered by tests.
- README examples use only contract-approved names and behaviors.
- Package metadata, examples, and generated docs are aligned with the first public release contract.
