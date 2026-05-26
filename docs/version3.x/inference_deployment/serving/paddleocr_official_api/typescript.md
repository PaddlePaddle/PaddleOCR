---
comments: true
---

# PaddleOCR 官方 API TypeScript SDK

TypeScript SDK 面向 Node.js 18 及以上环境，调用 PaddleOCR 官方 API 完成 OCR 与文档解析任务。它使用官方托管服务，不运行本地 PaddleOCR 推理。

## 安装与认证

请先在 [AI Studio Access Token 页面](https://aistudio.baidu.com/account/accessToken) 获取访问令牌。

```bash
npm install @paddleocr/api-sdk
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

客户端默认读取 `PADDLEOCR_ACCESS_TOKEN`，也可以传入 `token`：

```ts
import { PaddleOCRClient } from "@paddleocr/api-sdk";

const client = new PaddleOCRClient({
  token: process.env.PADDLEOCR_ACCESS_TOKEN,
});
```

## 快速开始

```ts
import { Model, PaddleOCRClient } from "@paddleocr/api-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  fileUrl: "https://example.com/invoice.pdf",
  model: Model.PPOCRv5,
});
console.log(result.jobId, result.pages.length);
```

本地文件使用 `filePath`。`fileUrl` 与 `filePath` 必须二选一。

## 公共 API

TypeScript SDK 常用公共方法包括：

- `ocr(...)`：提交 OCR 任务，等待完成并返回 OCR 结果。
- `parseDocument(...)`：提交文档解析任务，等待完成并返回文档解析结果。
- `submitOcr(...)`：只提交 OCR 任务，返回任务对象。
- `submitDocumentParsing(...)`：只提交文档解析任务，返回任务对象。
- `getStatus(jobId)`：执行一次非阻塞状态查询。
- `waitOcrResult(job)`：等待 OCR 任务完成并解析结果。
- `waitDocumentParsingResult(job)`：等待文档解析任务完成并解析结果。
- `saveResource(resourceUrl, destination, options)`：保存单个资源 URL。
- `saveOcrResultResources(result, destination, options)`：保存 OCR 结果对象引用的资源。
- `saveDocumentParsingResultResources(result, destination, options)`：保存文档解析结果对象引用的资源。

## 超时

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` 限制一次 HTTP 请求，包括提交、查询状态和下载资源。`pollTimeout` 限制 `ocr`、`parseDocument`、`waitOcrResult` 与 `waitDocumentParsingResult` 的总等待时间。公共方法还可以接收 `AbortSignal` 以便上层主动取消。

## 模型选择

表中的 `Model` 枚举是官方 API 模型名字符串的类型安全写法，提交请求时会转换为对应的实际模型名。也可以直接传入官方 API 模型名字符串，例如 `model: "PaddleOCR-VL-1.6"`。

| 任务 | 适用接口 | 默认模型 | 可选模型 | 参数类型 |
| --- | --- | --- | --- | --- |
| OCR | `ocr`、`submitOcr`、`waitOcrResult` | `Model.PPOCRv5` | `Model.PPOCRv5` | `OCROptions` |
| 文档解析 | `parseDocument`、`submitDocumentParsing`、`waitDocumentParsingResult` | `Model.PaddleOCRVL16` | `Model.PPStructureV3`、`Model.PaddleOCRVL`、`Model.PaddleOCRVL15`、`Model.PaddleOCRVL16` | 选择 `PPStructureV3` 时传入 `PPStructureV3Options`；选择 PaddleOCR-VL 系列模型时传入 `PaddleOCRVLOptions`。 |

常用对应关系：`Model.PPOCRv5` 对应 `PP-OCRv5`，`Model.PPStructureV3` 对应 `PP-StructureV3`，`Model.PaddleOCRVL` 对应 `PaddleOCR-VL`，`Model.PaddleOCRVL15` 对应 `PaddleOCR-VL-1.5`，`Model.PaddleOCRVL16` 对应 `PaddleOCR-VL-1.6`。

## 错误与资源保存

所有 SDK 错误都继承自 `PaddleOCRAPIError`，常见类型包括 `AuthError`、`InvalidRequestError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError`。

`saveResource` 用于单个资源 URL；如果要保存结果对象中的全部资源，请使用 `saveOcrResultResources` 或 `saveDocumentParsingResultResources`。
