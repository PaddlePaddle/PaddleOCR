---
comments: true
---

# PaddleOCR 官方 API Go SDK

Go SDK 调用 PaddleOCR 官方 API，把 OCR 或文档解析任务提交到官方托管服务。它不运行本地 PaddleOCR 推理，也不加载本地模型。

## 安装与认证

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

`NewClient` 默认读取 `PADDLEOCR_ACCESS_TOKEN`，也可以通过 `WithToken` 显式传入 token。

## 快速开始

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

本地文件使用 `FilePath`。`FileURL` 与 `FilePath` 必须二选一。

## 公共 API

Go SDK 的最终公共方法包括：

- `OCR(...)`：提交 OCR 任务，等待完成并返回 OCR 结果。
- `ParseDocument(...)`：提交文档解析任务，等待完成并返回文档解析结果。
- `SubmitOCR(...)`：只提交 OCR 任务，返回任务对象。
- `SubmitDocumentParsing(...)`：只提交文档解析任务，返回任务对象。
- `GetStatus(ctx, jobID)`：执行一次非阻塞状态查询。
- `WaitOCRResult(ctx, job)`：等待 OCR 任务完成并解析结果。
- `WaitDocumentParsingResult(ctx, job)`：等待文档解析任务完成并解析结果。
- `SaveResource(...)`：保存单个资源 URL。
- `SaveOCRResultResources(...)`：保存 OCR 结果对象引用的资源。
- `SaveDocumentParsingResultResources(...)`：保存文档解析结果对象引用的资源。

## 超时

```go
client, err := paddleocr.NewClient(
	paddleocr.WithRequestTimeout(30*time.Second),
	paddleocr.WithPollTimeout(5*time.Minute),
)
```

`WithRequestTimeout` 限制一次 HTTP 请求，包括提交、查询状态和下载资源。`WithPollTimeout` 限制 `OCR`、`ParseDocument`、`WaitOCRResult` 与 `WaitDocumentParsingResult` 的总等待时间。调用方也可以通过 `context.Context` 取消请求。

## 模型扩展

OCR 的 `Model` 可省略，默认是 `PPOCRv5`。当前 PaddleOCR 官方 API 版本只开放 PP-OCRv5 作为 OCR 模型。文档解析的 `Model` 可省略，默认是 `PaddleOCRVL16`。SDK 通过 `IsOCRModel` 与 `IsDocumentParsingModel` 集中校验模型类型，未来模型可在模型分类处集中添加。

## 错误与资源保存

Go SDK 暴露可与 `errors.As` 配合使用的类型化错误，覆盖 `AuthError`、`InvalidRequestError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError` 等情况。

资源保存默认不覆盖已有文件，也不会向结果资源 URL 发送 PaddleOCR 官方 API 鉴权头。结果对象批量保存要求目标路径是已存在的目录。

更多源码相邻参考见 `api_sdk/go/README.md`。
