# PaddleOCR TypeScript SDK

[English](README.md) | 简体中文

面向 PaddleOCR 官方 API 的 TypeScript 客户端。它会把 OCR 和文档解析任务提交到 PaddleOCR 官方托管服务；不会在本地执行 OCR 推理。

正式用户文档：

- [TypeScript SDK](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md)
- [TypeScript SDK（英文）](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.en.md)

## 安装

```bash
npm install @paddleocr/api-sdk
```

该包遵循语义化版本，并作为公开 scoped npm 包发布。

本地开发：

```bash
npm install
npm run build
```

## 最小示例

设置 `PADDLEOCR_ACCESS_TOKEN`，或在构造客户端时传入 `token`：

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

```ts
import { Model, PaddleOCRClient } from "@paddleocr/api-sdk";

const client = new PaddleOCRClient();
const result = await client.ocr({
  model: Model.PPOCRv5,
  fileUrl: "https://example.com/invoice.pdf",
});

console.log(result.jobId, result.pages.length);
```

通过 `model: Model.PPOCRv6`（或 `"PP-OCRv6"`）可指定 PP-OCRv6 云端 OCR 模型。
使用 `Model.PPOCRv5Latin`（或 `"PP-OCRv5-latin"`）可指定 PP-OCRv5 拉丁语系云端 OCR 模型。

文档解析默认使用 PaddleOCR-VL-1.6：

```ts
const doc = await client.parseDocument({
  filePath: "./report.pdf",
  options: {
    useChartRecognition: true,
  },
});

console.log(doc.jobId, doc.pages.length);
```

## 构建与测试

```bash
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
```
