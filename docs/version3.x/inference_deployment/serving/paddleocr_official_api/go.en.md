---
comments: true
---

# PaddleOCR official API Go SDK

The Go SDK calls the PaddleOCR official API and submits OCR or document parsing jobs to hosted services. It does not run local PaddleOCR inference or load local models.

## Install And Authenticate

First obtain an access token from the [AI Studio Access Token page](https://aistudio.baidu.com/account/accessToken).

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

`NewClient` reads `PADDLEOCR_ACCESS_TOKEN` by default and also accepts an explicit token through `WithToken`.

## Quick Start

```go
client, err := paddleocr.NewClient()
if err != nil {
	return err
}

result, err := client.OCR(ctx, &paddleocr.OCRRequest{
	Model:   paddleocr.PPOCRv5,
	FileURL: "https://example.com/invoice.pdf",
})
if err != nil {
	return err
}
fmt.Println(result.JobID, len(result.Pages))
```

Use `FilePath` for a local file. Pass exactly one of `FileURL` or `FilePath`.

## Public API

Common Go public methods include:

- `OCR(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `ParseDocument(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `SubmitOCR(...)`: submit only an OCR job and return a job object.
- `SubmitDocumentParsing(...)`: submit only a document parsing job and return a job object.
- `GetStatus(ctx, jobID)`: perform one non-blocking status request.
- `WaitOCRResult(ctx, job)`: wait for an OCR job and parse its result.
- `WaitDocumentParsingResult(ctx, job)`: wait for a document parsing job and parse its result.
- `SaveResource(...)`: save one resource URL.
- `SaveOCRResultResources(...)`: save resources referenced by an OCR result object.
- `SaveDocumentParsingResultResources(...)`: save resources referenced by a document parsing result object.

## Choose Models

The model constants in the table are type-safe aliases for the official API model-name strings. They are serialized to the corresponding model name when the request is submitted. You can also pass the official API model-name string directly, for example `Model: "PaddleOCR-VL-1.6"`.

| Task | Interfaces | Default model | Supported models | Option type |
| --- | --- | --- | --- | --- |
| OCR | `OCR`, `SubmitOCR`, `WaitOCRResult` | `PPOCRv6` | `PPOCRv5`, `PPOCRv6` | `*OCROptions` |
| Document parsing | `ParseDocument`, `SubmitDocumentParsing`, `WaitDocumentParsingResult` | `PaddleOCRVL16` | `PPStructureV3`, `PaddleOCRVL`, `PaddleOCRVL15`, `PaddleOCRVL16` | Use `*PPStructureV3Options` with `PPStructureV3`, and `*PaddleOCRVLOptions` with PaddleOCR-VL models. |

## Configuration

### Client Configuration

```go
client, err := paddleocr.NewClient(
	paddleocr.WithRequestTimeout(30*time.Second),
	paddleocr.WithPollTimeout(5*time.Minute),
)
```

`WithRequestTimeout` limits one HTTP request, including submit, status, and resource downloads. `WithPollTimeout` limits the total wait time for `OCR`, `ParseDocument`, `WaitOCRResult`, and `WaitDocumentParsingResult`. Callers can also cancel requests through `context.Context`.

Override the service base URL via the `PADDLEOCR_BASE_URL` environment variable or `WithBaseURL`:

```go
client, err := paddleocr.NewClient(
    paddleocr.WithBaseURL("https://my-proxy.com/paddle"),
)
```

Inject a custom `*http.Client` via `WithHTTPClient` for proxies, custom TLS, or retry policies:

```go
client, err := paddleocr.NewClient(
    paddleocr.WithHTTPClient(myHTTPClient),
)
```

### Request Options

Go SDK Options struct fields use PascalCase, serialized to camelCase automatically. Pointer fields set to `nil` are omitted. See the struct source or Official API Reference for complete field definitions.

#### OCROptions (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `UseDocOrientationClassify` | *bool | Document orientation classification |
| `UseDocUnwarping` | *bool | Document unwarping |
| `Visualize` | *bool | Return visualization images |

#### PPStructureV3Options (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `UseTableRecognition` | *bool | Table recognition |
| `UseFormulaRecognition` | *bool | Formula recognition |
| `UseChartRecognition` | *bool | Chart recognition |
| `PrettifyMarkdown` | *bool | Markdown prettification |

#### PaddleOCRVLOptions (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `UseLayoutDetection` | *bool | Layout detection |
| `UseChartRecognition` | *bool | Chart recognition |
| `Temperature` | *float64 | Sampling temperature |
| `PrettifyMarkdown` | *bool | Markdown prettification |

## Error Handling

The Go SDK exposes typed errors compatible with `errors.As`, including `AuthError`, `InvalidRequestError`, `RateLimitError`, `ServiceUnavailableError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

## Official API Reference

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Dmh4onssk)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/7mfz6dgx9)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/Vmkz2nz1p)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/fml7mozw5)

## Quota and Error Codes

- [API Quota Rules and Error Code Description](https://ai.baidu.com/ai-doc/AISTUDIO/pmjcld5qm)
