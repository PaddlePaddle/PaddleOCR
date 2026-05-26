---
comments: true
---

# PaddleOCR 官方 API CLI

`paddleocr api` 是 PaddleOCR CLI 中调用 PaddleOCR 官方 API 的子命令。它把文件 URL 或本地文件提交到官方托管服务，等待任务完成并输出 JSON；它不运行本地推理。

## 认证

CLI 默认读取 `PADDLEOCR_ACCESS_TOKEN`：

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

也可以使用 `--token` 显式传入 token。

## 基本用法

```bash
paddleocr api \
  --model_type ocr \
  --file_url https://example.com/invoice.pdf
```

`--model_type` 必填，可选值为 `ocr` 或 `document_parsing`。`--file_url` 与 `--file_path` 必须二选一。

## 常用参数

- `--model_type`：任务类型，`ocr` 或 `document_parsing`。
- `--model`：模型名称。OCR 任务默认使用 PP-OCRv5；文档解析任务未指定时默认使用 PaddleOCR-VL-1.5。模型会通过 PaddleOCR 官方 API SDK 的模型分类辅助函数校验。
- `--file_url`：待处理文件 URL。
- `--file_path`：待上传并处理的本地文件路径。
- `--request_timeout`：一次 HTTP 请求的超时时间，单位为秒。
- `--poll_timeout`：等待远端任务完成的总超时时间，单位为秒。
- `--output`：输出 JSON 文件路径；省略时打印到标准输出。
- `--page_ranges`：页码范围，例如 `2,4-6`。
- `--use_doc_orientation_classify`、`--use_doc_unwarping`、`--use_textline_orientation`：OCR 相关可选能力。
- `--use_chart_recognition`：文档解析相关可选能力。

## OCR 示例

```bash
paddleocr api \
  --model_type ocr \
  --model PP-OCRv5 \
  --file_path ./invoice.pdf \
  --request_timeout 300 \
  --poll_timeout 600 \
  --output ocr-result.json
```

## 文档解析示例

```bash
paddleocr api \
  --model_type document_parsing \
  --model PaddleOCR-VL-1.5 \
  --file_url https://example.com/report.pdf \
  --use_chart_recognition \
  --output doc-result.json
```

## 输出行为

命令成功时输出格式化 JSON。OCR 结果包含 `jobId` 和每页的 `prunedResult`、`ocrImageUrl`；文档解析结果包含 `jobId` 和每页的 `markdownText`、`markdownImages`、`outputImages`。如果指定 `--output`，CLI 写入该文件并打印保存位置；否则直接打印到标准输出。

错误会输出到标准错误并返回非零退出码。常见原因包括缺少 `PADDLEOCR_ACCESS_TOKEN`、模型与 `--model_type` 不匹配、请求超时、轮询超时、远端任务失败或响应格式异常。
