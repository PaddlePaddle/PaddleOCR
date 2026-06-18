---
comments: true
---

# PaddleOCR 官方 API Go SDK

Go SDK 调用 PaddleOCR 官方 API，把 OCR 或文档解析任务提交到官方托管服务。它不运行本地 PaddleOCR 推理，也不加载本地模型。

## 安装与认证

请先在 [AI Studio Access Token 页面](https://aistudio.baidu.com/account/accessToken) 获取访问令牌。

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
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

Go SDK 常用公共方法包括：

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

## 模型选择

表中的模型常量是官方 API 模型名字符串的类型安全写法，提交请求时会转换为对应的实际模型名。也可以直接传入官方 API 模型名字符串，例如 `Model: "PaddleOCR-VL-1.6"`。

| 任务 | 适用接口 | 默认模型 | 可选模型 | 参数类型 |
| --- | --- | --- | --- | --- |
| OCR | `OCR`、`SubmitOCR`、`WaitOCRResult` | `PPOCRv6` | `PPOCRv5`、`PPOCRv6` | `*OCROptions` |
| 文档解析 | `ParseDocument`、`SubmitDocumentParsing`、`WaitDocumentParsingResult` | `PaddleOCRVL16` | `PPStructureV3`、`PaddleOCRVL`、`PaddleOCRVL15`、`PaddleOCRVL16` | 选择 `PPStructureV3` 时传入 `*PPStructureV3Options`；选择 PaddleOCR-VL 系列模型时传入 `*PaddleOCRVLOptions`。 |

## 配置与参数

### 客户端配置

```go
client, err := paddleocr.NewClient(
	paddleocr.WithRequestTimeout(30*time.Second),
	paddleocr.WithPollTimeout(5*time.Minute),
)
```

`WithRequestTimeout` 限制一次 HTTP 请求，包括提交、查询状态和下载资源。`WithPollTimeout` 限制 `OCR`、`ParseDocument`、`WaitOCRResult` 与 `WaitDocumentParsingResult` 的总等待时间。调用方也可以通过 `context.Context` 取消请求。

通过环境变量 `PADDLEOCR_BASE_URL` 或 `WithBaseURL` 指定自定义服务地址：

```go
client, err := paddleocr.NewClient(
    paddleocr.WithBaseURL("https://my-proxy.com/paddle"),
)
```

通过 `WithHTTPClient` 注入自定义 `*http.Client`，适用于需要代理、自定义 TLS 或重试策略的场景：

```go
client, err := paddleocr.NewClient(
    paddleocr.WithHTTPClient(myHTTPClient),
)
```

### 请求参数

Go SDK 的 Options 结构体字段使用 PascalCase，序列化时自动转换为 camelCase。指针类型字段传 `nil` 表示不设置。完整字段定义见结构体源码或官方 API 参考。

#### OCROptions（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `UseDocOrientationClassify` | *bool | 文档方向分类 |
| `UseDocUnwarping` | *bool | 文档扭曲矫正 |
| `Visualize` | *bool | 是否返回可视化结果图 |

#### PPStructureV3Options（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `UseTableRecognition` | *bool | 表格识别 |
| `UseFormulaRecognition` | *bool | 公式识别 |
| `UseChartRecognition` | *bool | 图表识别 |
| `PrettifyMarkdown` | *bool | Markdown 美化 |

#### PaddleOCRVLOptions（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `UseLayoutDetection` | *bool | 版面检测 |
| `UseChartRecognition` | *bool | 图表识别 |
| `Temperature` | *float64 | 采样温度 |
| `PrettifyMarkdown` | *bool | Markdown 美化 |

## 错误处理

Go SDK 暴露可与 `errors.As` 配合使用的类型化错误，覆盖 `AuthError`、`InvalidRequestError`、`RateLimitError`、`ServiceUnavailableError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError` 等情况。

## 官方 API 参考

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Kmfl2ycs0)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/Fmfz6oh2e)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/2mh4okm66)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Cmkz2m0ma)

## 配额与错误码

- [API 配额规则和错误码说明](https://ai.baidu.com/ai-doc/AISTUDIO/Xmjclapam)
