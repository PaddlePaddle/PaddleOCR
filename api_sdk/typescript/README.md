# PaddleOCR TypeScript SDK

English | [简体中文](README_cn.md)

TypeScript client for the PaddleOCR official API. It submits OCR and document
parsing jobs to hosted PaddleOCR services; it does not perform local OCR
inference.

Official user docs:

- [TypeScript SDK](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md)
- [TypeScript SDK (English)](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.en.md)

## Install

```bash
npm install @paddleocr/api-sdk
```

This package follows SemVer and is published as a public scoped npm package.

For local development:

```bash
npm install
npm run build
```

## Minimal Usage

Set `PADDLEOCR_ACCESS_TOKEN` or pass `token` to the client:

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

Document parsing defaults to PaddleOCR-VL-1.6:

```ts
const doc = await client.parseDocument({
  filePath: "./report.pdf",
  options: {
    useChartRecognition: true,
  },
});

console.log(doc.jobId, doc.pages.length);
```

## Build And Test

```bash
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
```
