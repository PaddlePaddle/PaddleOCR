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

## Timeouts

```go
client, err := paddleocr.NewClient(
	paddleocr.WithRequestTimeout(30*time.Second),
	paddleocr.WithPollTimeout(5*time.Minute),
)
```

`WithRequestTimeout` limits one HTTP request, including submit, status, and resource downloads. `WithPollTimeout` limits the total wait time for `OCR`, `ParseDocument`, `WaitOCRResult`, and `WaitDocumentParsingResult`. Callers can also cancel requests through `context.Context`.

## Choose Models

| Task | Interfaces | Default model | Supported models | Option type |
| --- | --- | --- | --- | --- |
| OCR | `OCR`, `SubmitOCR`, `WaitOCRResult` | `PPOCRv5` | `PPOCRv5` | `*OCROptions` |
| Document parsing | `ParseDocument`, `SubmitDocumentParsing`, `WaitDocumentParsingResult` | `PaddleOCRVL16` | `PPStructureV3`, `PaddleOCRVL`, `PaddleOCRVL15`, `PaddleOCRVL16` | Use `*PPStructureV3Options` with `PPStructureV3`, and `*PaddleOCRVLOptions` with PaddleOCR-VL models. |

## Errors And Resource Saving

The Go SDK exposes typed errors compatible with `errors.As`, including `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

Resource saving supports one resource URL or all resources referenced by a result object. Result-object bulk saving requires the destination to be an existing directory.
