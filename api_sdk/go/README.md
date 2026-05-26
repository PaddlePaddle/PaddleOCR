# PaddleOCR Go SDK

Go client for the PaddleOCR official API. This SDK submits OCR and document parsing
jobs to hosted PaddleOCR services; it does not run local PaddleOCR inference or
load local models.

This file is a source-adjacent reference for Go SDK maintenance. The official user
docs live in `docs/version3.x/api_sdk/go.md` and
`docs/version3.x/api_sdk/go.en.md`.

## Install

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
```

## Authentication

Set `PADDLEOCR_ACCESS_TOKEN` or pass `WithToken` when constructing the client.

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

```go
client, err := paddleocr.NewClient(paddleocr.WithToken("token"))
if err != nil {
	return err
}
```

`NewClient` returns `AuthError` if no token is available.

## Client Options

```go
client, err := paddleocr.NewClient(
	paddleocr.WithToken("token"),
	paddleocr.WithBaseURL("https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"),
	paddleocr.WithRequestTimeout(30*time.Second),
	paddleocr.WithPollTimeout(5*time.Minute),
)
```

`WithRequestTimeout` limits one HTTP request, including submit, status, and
resource download calls. `WithPollTimeout` limits the total wait loop for
`OCR`, `ParseDocument`, `WaitOCRResult`, `WaitDocumentParsingResult`,
`Operation.WaitOCR`, and `Operation.WaitDocumentParsing`.

## OCR

URL OCR:

```go
result, err := client.OCR(ctx, &paddleocr.OCRRequest{
	Model:   paddleocr.PPOCRv5,
	FileURL: "https://example.com/invoice.pdf",
})
if err != nil {
	return err
}
fmt.Println(result.JobID, len(result.Pages))
```

File OCR:

```go
result, err := client.OCR(ctx, &paddleocr.OCRRequest{
	FilePath:   "./invoice.pdf",
	PageRanges: "1-3",
	Options: &paddleocr.OCROptions{
		UseTextlineOrientation: paddleocr.Bool(true),
	},
})
```

Pass exactly one of `FileURL` or `FilePath`.

`Model` is optional for OCR and defaults to `PPOCRv5`. PP-OCRv5 is the only OCR
model supported by the current PaddleOCR official API release; model validation
is centralized in `IsOCRModel` and `IsDocumentParsingModel` so future OCR models
can be added without changing submit/wait logic.

Document parsing `Model` is optional and defaults to `PaddleOCRVL15`.

## Document Parsing

```go
result, err := client.ParseDocument(ctx, &paddleocr.DocParsingRequest{
	FilePath: "./sample.pdf",
	Options: &paddleocr.DocParsingOptions{
		UseChartRecognition: paddleocr.Bool(true),
	},
})
if err != nil {
	return err
}
fmt.Println(result.JobID, len(result.Pages))
```

Supported document parsing models include `PPStructureV3`, `PaddleOCRVL`, and
`PaddleOCRVL15`.

## Submit, Status, And Wait

Convenience methods submit, wait, fetch, and parse in one call:

```go
ocrResult, err := client.OCR(ctx, &paddleocr.OCRRequest{
	FileURL: "https://example.com/a.pdf",
})
docResult, err := client.ParseDocument(ctx, &paddleocr.DocParsingRequest{
	Model:   paddleocr.PaddleOCRVL15,
	FileURL: "https://example.com/b.pdf",
})
```

Use explicit asynchronous control when you need status checks or concurrent
waits:

```go
ocrOp, err := client.SubmitOCR(ctx, &paddleocr.OCRRequest{
	FileURL: "https://example.com/a.pdf",
})
docOp, err := client.SubmitDocumentParsing(ctx, &paddleocr.DocParsingRequest{
	Model:    paddleocr.PaddleOCRVL15,
	FilePath: "./b.pdf",
})

status, err := client.GetStatus(ctx, ocrOp.JobID)
fmt.Println(status.State, status.Progress)

ocrResult, err := client.WaitOCRResult(ctx, &ocrOp.Job)
docResult, err := client.WaitDocumentParsingResult(ctx, &docOp.Job)
```

`GetStatus(ctx, jobID)` performs one non-blocking status request. It does not
wait for completion or download result payloads.

`SubmitOCR` and `SubmitDocumentParsing` return `Operation` values with
convenience wait and poll methods:

```go
status, done, err := ocrOp.Poll(ctx)
result, err := ocrOp.WaitOCR(ctx)
docResult, err := docOp.WaitDocumentParsing(ctx)
```

## Save Resources

Result resource URLs can be saved with `SaveResource`. Result objects can be
saved in bulk with `SaveOCRResultResources` or
`SaveDocumentParsingResultResources`. These helpers do not send API
authorization headers to resource URLs and will not overwrite existing files
unless requested.

```go
savedPath, err := client.SaveResource(
	ctx,
	imageURL,
	"./outputs",
	paddleocr.WithOverwrite(true),
)
if err != nil {
	return err
}
fmt.Println(savedPath)
```

If the destination is an existing directory, the filename is derived from the
resource URL. For result-object bulk saving, the destination must be an existing
directory:

```go
savedOCRImages, err := client.SaveOCRResultResources(ctx, ocrResult, "./outputs")
if err != nil {
	return err
}

savedDocImages, err := client.SaveDocumentParsingResultResources(
	ctx,
	docResult,
	"./outputs",
	paddleocr.WithOverwrite(true),
)
if err != nil {
	return err
}
fmt.Println(savedOCRImages, savedDocImages)
```

OCR result images are saved as stable `ocr-page-{n}` filenames with safe URL
extensions when available. Document parsing images use
`DocParsingPage.MarkdownImages` and `DocParsingPage.OutputImages` map keys as
filenames after validation; keys must be non-empty and must not be absolute
paths, traversal markers, or contain path separators.

## Errors

The SDK exposes typed errors compatible with `errors.As`:

| Error | Meaning |
| --- | --- |
| `AuthError` | Missing token or HTTP 401/403 |
| `InvalidRequestError` | Invalid SDK input or HTTP 400 |
| `APIError` | Other non-2xx HTTP response |
| `NetworkError` | Transport failure before a response |
| `JobFailedError` | Remote job reached `failed` |
| `RequestTimeoutError` | One HTTP request exceeded `WithRequestTimeout` |
| `PollTimeoutError` | Wait loop exceeded `WithPollTimeout` |
| `FileNotFoundError` | Local file or destination parent is missing |
| `ResponseFormatError` | Successful API response is malformed |
| `ResultParseError` | Result JSONL cannot be parsed |

```go
var authErr *paddleocr.AuthError
if errors.As(err, &authErr) {
	log.Fatalf("authentication failed: %v", authErr)
}
```

## Build And Test

```bash
go test ./...
go vet ./...
go test -race ./...
```

`go test -race ./...` is recommended before public release.
