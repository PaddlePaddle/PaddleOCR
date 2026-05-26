---
comments: true
---

# PaddleOCR official API Go SDK

The Go SDK calls the PaddleOCR official API and submits OCR or document parsing jobs to hosted services. It does not run local PaddleOCR inference or load local models.

## Install And Authenticate

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
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

Document parsing example:

```go
result, err := client.ParseDocument(ctx, &paddleocr.DocParsingRequest{
	Model:    paddleocr.PaddleOCRVL15,
	FilePath: "./report.pdf",
	Options: &paddleocr.DocParsingOptions{
		UseChartRecognition: paddleocr.Bool(true),
	},
})
if err != nil {
	return err
}
fmt.Println(result.JobID, len(result.Pages))
```

## Public API

The final Go public methods are:

- `OCR(...)`: submit an OCR job, wait for completion, and return an OCR result.
- `ParseDocument(...)`: submit a document parsing job, wait for completion, and return a document parsing result.
- `SubmitOCR(...)`: submit only an OCR job and return an `Operation`.
- `SubmitDocumentParsing(...)`: submit only a document parsing job and return an `Operation`.
- `GetStatus(ctx, jobID)`: perform one non-blocking status request.
- `WaitOCRResult(ctx, job)`: wait for an OCR job and parse its result.
- `WaitDocumentParsingResult(ctx, job)`: wait for a document parsing job and parse its result.
- `SaveResource(...)`: save one resource URL.
- `SaveOCRResultResources(...)`: save resources referenced by an OCR result object.
- `SaveDocumentParsingResultResources(...)`: save resources referenced by a document parsing result object.

`Operation` also provides task-local poll and wait methods.

## Timeouts

```go
client, err := paddleocr.NewClient(
	paddleocr.WithRequestTimeout(30*time.Second),
	paddleocr.WithPollTimeout(5*time.Minute),
)
```

`WithRequestTimeout` limits one HTTP request, including submit, status, and resource downloads. `WithPollTimeout` limits the total wait time for `OCR`, `ParseDocument`, `WaitOCRResult`, `WaitDocumentParsingResult`, and `Operation` wait methods. Callers can also cancel requests through `context.Context`.

## Model Extensibility

The OCR `Model` is optional and defaults to `PPOCRv5`. PP-OCRv5 is the only OCR model exposed by the current PaddleOCR official API release.

Document parsing `Model` is optional and defaults to `PaddleOCRVL15`. Supported document parsing models include `PPStructureV3`, `PaddleOCRVL`, and `PaddleOCRVL15`. The SDK validates model categories through `IsOCRModel` and `IsDocumentParsingModel`, so future models can be added centrally.

## Errors And Resource Saving

The Go SDK exposes typed errors compatible with `errors.As`, including `AuthError`, `InvalidRequestError`, `APIError`, `NetworkError`, `JobFailedError`, `RequestTimeoutError`, `PollTimeoutError`, `ResponseFormatError`, and `ResultParseError`.

Resource saving does not overwrite existing files by default and does not send PaddleOCR official API authorization headers to result resource URLs. Result-object bulk saving requires the destination to be an existing directory.

See the [api_sdk/go/README.md](../../../api_sdk/go/README.md) source-adjacent reference.
