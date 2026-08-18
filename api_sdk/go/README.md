# PaddleOCR Go SDK

English | [简体中文](README_cn.md)

Go client for the PaddleOCR official API. It submits OCR and document parsing
jobs to hosted PaddleOCR services; it does not run local PaddleOCR inference or
load local models.

Official user docs:

- [Go SDK](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md)
- [Go SDK (English)](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.en.md)

## Install

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
```

Versioned releases use submodule tags such as `api_sdk/go/v0.1.0`.

## Minimal Usage

Set `PADDLEOCR_ACCESS_TOKEN` or pass `WithToken` when constructing the client:

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

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

Set `Model: paddleocr.PPOCRv6` (or `"PP-OCRv6"`) to use the PP-OCRv6 hosted OCR model.
Set `Model: paddleocr.PPOCRv5Latin` (or `"PP-OCRv5-latin"`) to use the PP-OCRv5 Latin-script hosted OCR model.

Document parsing defaults to PaddleOCR-VL-1.6:

```go
doc, err := client.ParseDocument(ctx, &paddleocr.DocParsingRequest{
	FilePath: "./report.pdf",
	Options: &paddleocr.PaddleOCRVLOptions{
		UseChartRecognition: paddleocr.Bool(true),
	},
})
if err != nil {
	return err
}
fmt.Println(doc.JobID, len(doc.Pages))
```

## Build And Test

```bash
go test ./...
go vet ./...
go test -race ./...
```

`go test -race ./...` is recommended before public release.
