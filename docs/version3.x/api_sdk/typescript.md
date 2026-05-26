---
comments: true
---

# PaddleOCR 官方 API TypeScript SDK

TypeScript SDK 面向 Node.js 18 及以上环境，调用 PaddleOCR 官方 API 完成 OCR 与文档解析任务。它使用官方托管服务，不运行本地 PaddleOCR 推理。

## 安装与认证

```bash
npm install paddleocr-sdk
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

客户端默认读取 `PADDLEOCR_ACCESS_TOKEN`，也可以传入 `token`：

```ts
import { PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient({
  token: process.env.PADDLEOCR_ACCESS_TOKEN,
});
```

## 快速开始

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  fileUrl: "https://example.com/invoice.pdf",
  model: Model.PPOCRv5,
});
console.log(result.jobId, result.pages.length);
```

本地文件使用 `filePath`。`fileUrl` 与 `filePath` 必须二选一。

文档解析示例：

```ts
import { Model, PaddleOCRClient } from "paddleocr-sdk";

const client = new PaddleOCRClient();
const result = await client.parseDocument({
  model: Model.PaddleOCRVL15,
  filePath: "./report.pdf",
  options: { useChartRecognition: true },
});
console.log(result.jobId, result.pages.length);
```

## 公共 API

TypeScript SDK 的最终公共方法包括：

- `ocr(...)`：提交 OCR 任务，等待完成并返回 OCR 结果。
- `parseDocument(...)`：提交文档解析任务，等待完成并返回文档解析结果。
- `submitOcr(...)`：只提交 OCR 任务，返回任务对象。
- `submitDocumentParsing(...)`：只提交文档解析任务，返回任务对象。
- `getStatus(jobId)`：执行一次非阻塞状态查询。
- `waitOcrResult(job)`：等待 OCR 任务完成并解析结果。
- `waitDocumentParsingResult(job)`：等待文档解析任务完成并解析结果。
- `saveResource(resource, destination, options)`：保存单个资源 URL 或结果对象中的资源。

## 超时

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` 限制一次 HTTP 请求，包括提交、查询状态和下载资源。`pollTimeout` 限制 `ocr`、`parseDocument`、`waitOcrResult` 与 `waitDocumentParsingResult` 的总等待时间。公共方法还可以接收 `AbortSignal` 以便上层主动取消。

## 模型扩展

OCR 的 `model` 可省略，默认是 `Model.PPOCRv5`。当前 PaddleOCR 官方 API 版本只开放 PP-OCRv5 作为 OCR 模型。

文档解析的 `model` 可选，默认使用 `Model.PaddleOCRVL15`。支持的文档解析模型包括 `Model.PPStructureV3`、`Model.PaddleOCRVL` 和 `Model.PaddleOCRVL15`。SDK 通过 `isOCRModel` 与 `isDocumentParsingModel` 集中校验模型类型，未来模型可在模型分类处集中添加。

## 错误与资源保存

所有 SDK 错误都继承自 `PaddleOCRAPIError`，常见类型包括 `AuthError`、`InvalidRequestError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError`。

`saveResource` 支持保存单个资源 URL，也支持保存 `OCRResult` 或 `DocParsingResult` 中的全部资源。默认不覆盖已有文件，资源下载也不会向结果资源 URL 发送 PaddleOCR 官方 API 鉴权头。

更多源码相邻参考见 [api_sdk/typescript/README.md](../../../api_sdk/typescript/README.md)。
