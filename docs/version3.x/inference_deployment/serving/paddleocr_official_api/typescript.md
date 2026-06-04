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

## 模型选择

表中的 `Model` 枚举是官方 API 模型名字符串的类型安全写法，提交请求时会转换为对应的实际模型名。也可以直接传入官方 API 模型名字符串，例如 `model: "PaddleOCR-VL-1.6"`。

| 任务 | 适用接口 | 默认模型 | 可选模型 | 参数类型 |
| --- | --- | --- | --- | --- |
| OCR | `ocr`、`submitOcr`、`waitOcrResult` | `Model.PPOCRv5` | `Model.PPOCRv5` | `OCROptions` |
| 文档解析 | `parseDocument`、`submitDocumentParsing`、`waitDocumentParsingResult` | `Model.PaddleOCRVL16` | `Model.PPStructureV3`、`Model.PaddleOCRVL`、`Model.PaddleOCRVL15`、`Model.PaddleOCRVL16` | 选择 `PPStructureV3` 时传入 `PPStructureV3Options`；选择 PaddleOCR-VL 系列模型时传入 `PaddleOCRVLOptions`。 |

## 配置与参数

### 客户端配置

```ts
const client = new PaddleOCRClient({
  requestTimeout: 300_000,
  pollTimeout: 600_000,
});
```

`requestTimeout` 限制一次 HTTP 请求，包括提交、查询状态和下载资源。`pollTimeout` 限制 `ocr`、`parseDocument`、`waitOcrResult` 与 `waitDocumentParsingResult` 的总等待时间。公共方法还可以接收 `AbortSignal` 以便上层主动取消。

通过环境变量 `PADDLEOCR_BASE_URL` 或 `baseUrl` 参数指定自定义服务地址：

```ts
const client = new PaddleOCRClient({
  baseUrl: "https://my-proxy.com/paddle",
});
```

通过 `fetch` 选项注入自定义 fetch 实现，适用于需要代理或自定义网络层的场景：

```ts
const client = new PaddleOCRClient({
  fetch: myCustomFetch,
});
```

### 请求参数

TypeScript SDK 的参数名使用 camelCase，与官方 API 字段名一致。未设置的字段不会发送，使用服务端默认值。完整字段定义见接口源码或官方 API 参考。

#### OCROptions（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `useDocOrientationClassify` | boolean | 文档方向分类 |
| `useDocUnwarping` | boolean | 文档扭曲矫正 |
| `visualize` | boolean | 是否返回可视化结果图 |

#### PPStructureV3Options（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `useTableRecognition` | boolean | 表格识别 |
| `useFormulaRecognition` | boolean | 公式识别 |
| `useChartRecognition` | boolean | 图表识别 |
| `prettifyMarkdown` | boolean | Markdown 美化 |

#### PaddleOCRVLOptions（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `useLayoutDetection` | boolean | 版面检测 |
| `useChartRecognition` | boolean | 图表识别 |
| `temperature` | number | 采样温度 |
| `prettifyMarkdown` | boolean | Markdown 美化 |

## 错误处理

所有 SDK 错误都继承自 `PaddleOCRAPIError`，常见类型包括 `AuthError`、`InvalidRequestError`、`RateLimitError`、`ServiceUnavailableError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError`。

## 官方 API 参考

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Kmfl2ycs0)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/Fmfz6oh2e)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/2mh4okm66)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Cmkz2m0ma)

## 配额与错误码

- [API 配额规则和错误码说明](https://ai.baidu.com/ai-doc/AISTUDIO/Xmjclapam)
