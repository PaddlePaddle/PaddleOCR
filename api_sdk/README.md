# PaddleOCR official API SDKs

English | [简体中文](README_cn.md)

This directory contains source-adjacent maintainer files for the PaddleOCR
official API SDKs. The SDKs call hosted PaddleOCR official API services; they do
not run local PaddleOCR inference or load local models.

The official user documentation:

- [Overview](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/overview.md)
- [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md)
- [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md)
- [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md)
- [CLI](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/cli.md)

## Maintainer Files

| File | Purpose |
| --- | --- |
| [`typescript/README.md`](typescript/README.md) | Package-level README for the TypeScript SDK. |
| [`go/README.md`](go/README.md) | Package-level README for the Go SDK. |

The Python SDK is part of the main `paddleocr` package.

## Package Locations

| Language | Source location | User docs |
| --- | --- | --- |
| Python | [`../paddleocr`](../paddleocr) | [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md) |
| TypeScript | [`typescript`](typescript) | [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md) |
| Go | [`go`](go) | [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md) |

## Validation

Run these from the PaddleOCR repo root directory:

```bash
# Python
python -m pytest tests/api_client/

# TypeScript
cd api_sdk/typescript
npm run lint
npm test

# Go
cd ../go
go test ./...
```
