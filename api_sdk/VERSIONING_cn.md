# API SDK 版本控制

[English](VERSIONING.md) | 简体中文

PaddleOCR 官方 API SDK 遵循语义化版本。首个稳定公开版本之前，SDK 可以使用
`0.x` 版本，此时公共 API 仍可能继续收敛。发布 `1.0.0` 之后，破坏性变更必须提升主版本号。

## TypeScript

TypeScript SDK 以 npm 包 `@paddleocr/api-sdk` 发布。

- 源码：[`typescript`](typescript)
- 版本来源：[`typescript/package.json`](typescript/package.json)
- 发布产物：npm 包
- 包访问级别：公开 scoped package

推荐发布流程：

```bash
cd typescript
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
npm version <patch|minor|major>
npm publish --access public
```

兼容性 bug fix 使用 `patch`，向后兼容的新功能使用 `minor`，破坏性变更使用
`major`。每次发布包版本时，都应同步更新 `CHANGELOG.md`。

## Go

Go SDK 是仓库子目录 `api_sdk/go` 下的 Go module。

- Module path：`github.com/PaddlePaddle/PaddleOCR/api_sdk/go`
- 版本来源：Git tags
- 发布产物：可通过 `go get` 解析的 Go module 版本

由于该 module 位于 monorepo 子目录中，tag 必须带上 module 路径前缀：

```bash
git tag api_sdk/go/v0.1.0
git push origin api_sdk/go/v0.1.0
```

用户可通过以下命令安装指定版本：

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go@v0.1.0
```

如果 Go SDK 进入 v2 或更高主版本，需遵循 Go module 规则，在 module path 中加入主版本后缀，例如：

```go
module github.com/PaddlePaddle/PaddleOCR/api_sdk/go/v2
```

对应 tag 仍应使用子目录前缀，例如 `api_sdk/go/v2.0.0`。

## Changelog

使用 [`CHANGELOG_cn.md`](CHANGELOG_cn.md) 维护影响任一 SDK 的发布说明。每个发布条目中可按
TypeScript、Go、Python 分组记录语言特定变更。
