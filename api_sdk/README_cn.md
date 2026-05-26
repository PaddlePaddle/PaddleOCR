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

Python SDK 是主 `paddleocr` 包的一部分，公共入口由 `paddleocr` 导出；私有实现包不是受支持的导入路径。因此本目录不再维护单独的 Python 包级 README。

## 包位置

| 语言 | 源码位置 | 用户文档 |
| --- | --- | --- |
| Python | [`../paddleocr`](../paddleocr) | [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md) |
| TypeScript | [`typescript`](typescript) | [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md) |
| Go | [`go`](go) | [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md) |

## 验证

除非命令显式切换目录，否则请在本目录执行：

```bash
cd typescript
npm run lint
npm test

cd ../go
go test ./...

cd ..
git -C .. diff --check
git status --short --ignored=matching -- typescript
```

TypeScript SDK 使用 npm 包名 `@paddleocr/api-sdk`。Go 的版本化发布应使用
`api_sdk/go/v0.1.0` 这类子目录 module tag。
