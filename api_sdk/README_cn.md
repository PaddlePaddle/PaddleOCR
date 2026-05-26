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
| [`CONTRACT_cn.md`](CONTRACT_cn.md) | 首个公开版本的跨语言公共 API 契约。 |
| [`VERSIONING_cn.md`](VERSIONING_cn.md) | SemVer、npm 包与 Go module tag 策略。 |
| [`CHANGELOG_cn.md`](CHANGELOG_cn.md) | 面向 SDK 用户可见变更的发布说明。 |
| [`RELEASE_CHECKLIST_cn.md`](RELEASE_CHECKLIST_cn.md) | 发布前验证命令与人工检查项。 |
| [`typescript/README_cn.md`](typescript/README_cn.md) | TypeScript SDK 的包级 README。 |
| [`go/README_cn.md`](go/README_cn.md) | Go SDK 的包级 README。 |

Python SDK 源码是主 `paddleocr` 包的一部分，位于
[`../paddleocr/_api_client`](../paddleocr/_api_client)，因此本目录不再维护单独的
Python 包级 README。

## 包位置

| 语言 | 源码位置 | 用户文档 |
| --- | --- | --- |
| Python | [`../paddleocr/_api_client`](../paddleocr/_api_client) | [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md) |
| TypeScript | [`typescript`](typescript) | [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md) |
| Go | [`go`](go) | [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md) |

所有 SDK 默认读取 `PADDLEOCR_ACCESS_TOKEN`。OCR 默认使用 PP-OCRv5；文档解析默认使用 PaddleOCR-VL-1.6。详细用法示例应放在用户文档中，而不是本目录。

## 验证

除非命令显式切换目录，否则请在本目录执行：

```bash
python -m pytest ../tests/test_api_client -q

cd typescript
npm run lint
npm test

cd ../go
go test ./...

cd ..
git -C .. diff --check
git status --short --ignored=matching -- typescript
```

公开发布前请执行 [`RELEASE_CHECKLIST_cn.md`](RELEASE_CHECKLIST_cn.md) 中的扩展检查。
