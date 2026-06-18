# PaddleOCR 官方 API SDK

[English](README.md) | 简体中文

本目录包含 PaddleOCR 官方 API SDK 的源码相邻维护文档。SDK 调用 PaddleOCR 官方 API 托管服务；它们不在本地执行 PaddleOCR 推理，也不加载本地模型。

正式用户文档：

- [总览](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/overview.md)
- [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md)
- [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md)
- [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md)
- [CLI](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/cli.md)

## 维护者文件

| 文件 | 作用 |
| --- | --- |
| [`typescript/README_cn.md`](typescript/README_cn.md) | TypeScript SDK 的包级 README。 |
| [`go/README_cn.md`](go/README_cn.md) | Go SDK 的包级 README。 |

Python SDK 是主 `paddleocr` 包的一部分。

## 包位置

| 语言 | 源码位置 | 用户文档 |
| --- | --- | --- |
| Python | [`../paddleocr`](../paddleocr) | [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md) |
| TypeScript | [`typescript`](typescript) | [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md) |
| Go | [`go`](go) | [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md) |

## 验证

在 PaddleOCR 仓库根目录执行：

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
