# PaddleOCR Go SDK

[English](README.md) | 简体中文

面向 PaddleOCR 官方 API 的 Go 客户端。它会把 OCR 和文档解析任务提交到 PaddleOCR 官方托管服务；不会运行本地 PaddleOCR 推理，也不会加载本地模型。

正式用户文档：

- [Go SDK](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md)
- [Go SDK（英文）](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.en.md)

## 安装

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
```

版本化发布使用 `api_sdk/go/v0.1.0` 这类子目录 module tag。

## 最小示例

设置 `PADDLEOCR_ACCESS_TOKEN`，或在构造客户端时传入 `WithToken`：

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

将 `Model` 设为 `paddleocr.PPOCRv6`（或 `"PP-OCRv6"`）可使用 PP-OCRv6 云端 OCR 模型。
将 `Model` 设为 `paddleocr.PPOCRv5Latin`（或 `"PP-OCRv5-latin"`）可使用 PP-OCRv5 拉丁语系云端 OCR 模型。

文档解析默认使用 PaddleOCR-VL-1.6：

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

## 构建与测试

```bash
go test ./...
go vet ./...
go test -race ./...
```

公开发布前建议运行 `go test -race ./...`。
