# API SDK 更新日志

[English](CHANGELOG.md) | 简体中文

这里记录 PaddleOCR 官方 API SDK 的重要变更。

本项目遵循语义化版本。包版本与 tag 规则见 [`VERSIONING_cn.md`](VERSIONING_cn.md)。

## 未发布

- 将 TypeScript 包名改为 `@paddleocr/api-sdk`，与 PaddleOCR npm package
  namespace 保持一致，便于公开发布。
- 明确 Go 子目录 module 使用 `api_sdk/go/vX.Y.Z` 格式的 tag。
- 为 TypeScript 包发布补充发布元数据与验证门禁。
