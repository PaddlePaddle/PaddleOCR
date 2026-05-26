# API SDK 发布检查清单

[English](RELEASE_CHECKLIST.md) | 简体中文

本清单是 PaddleOCR 官方 API SDK 的发布门禁。除非命令显式切换目录，否则请在
`api_sdk` 目录执行。

## 文档检查

- 确认 [`CONTRACT_cn.md`](CONTRACT_cn.md) 与
  [`../docs/version3.x/inference_deployment/serving/paddleocr_official_api/`](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/)
  下的公开 API 文档一致。
- 确认 `api_sdk/` 中的文件仍是维护者文档或包级参考，而不是用户文档的第二份副本。
- 确认文档和示例使用 `PADDLEOCR_ACCESS_TOKEN`，文档解析默认模型为
  PaddleOCR-VL-1.6，并且只使用最终公开 API 名称。
- 确认 [`VERSIONING_cn.md`](VERSIONING_cn.md) 与
  [`CHANGELOG_cn.md`](CHANGELOG_cn.md) 已反映本次发布版本、npm 包名和 Go tag 策略。

## 验证命令

```bash
python -m pytest ../tests/test_api_client -q
python -c "import paddleocr; print(paddleocr.__version__)"

cd typescript
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
npm pack --dry-run

cd go
go test ./...
go vet ./...
go test -race ./...

git -C .. diff --check
git status --short --ignored=matching -- typescript
```

发布前请确认不会意外提交生成产物，除非包发布流程确实需要这些产物。
`go test -race ./...` 可在日常修复中跳过，但公开发布前应在本地平台支持时通过。

## 发布决策

仅在满足以下条件时发布：

- 上述验证命令通过；若存在例外，必须有明确负责人和发布说明。
- 公开示例可以针对打包后的 SDK 编译或通过类型检查。
- 没有意外跟踪 `node_modules`、`dist`、缓存或临时生成文件。
- TypeScript 包元数据指向 `@paddleocr/api-sdk`；Go 发布 tag 使用
  `api_sdk/go/vX.Y.Z` 子目录 module tag 格式。
- 所有延期事项都已由发布负责人接受，并记录在发布说明中。
