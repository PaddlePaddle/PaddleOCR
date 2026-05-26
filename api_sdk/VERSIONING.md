# API SDK Versioning

English | [简体中文](VERSIONING_cn.md)

The PaddleOCR official API SDKs follow Semantic Versioning. Before the first
stable public release, SDKs may use `0.x` versions while public API details are
still being finalized. After `1.0.0`, breaking changes require a new major
version.

## TypeScript

The TypeScript SDK is published as the npm package `@paddleocr/api-sdk`.

- Source: [`typescript`](typescript)
- Version source of truth: [`typescript/package.json`](typescript/package.json)
- Release artifact: npm package
- Package access: public scoped package

Recommended release flow:

```bash
cd typescript
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
npm version <patch|minor|major>
npm publish --access public
```

Use `patch` for compatible bug fixes, `minor` for backward-compatible features,
and `major` for breaking changes. Keep `CHANGELOG.md` updated whenever a package
version is released.

## Go

The Go SDK is a Go module under the repository subdirectory `api_sdk/go`.

- Module path: `github.com/PaddlePaddle/PaddleOCR/api_sdk/go`
- Version source of truth: Git tags
- Release artifact: Go module version resolved by `go get`

Because this is a subdirectory module in a monorepo, tags must include the module
path prefix:

```bash
git tag api_sdk/go/v0.1.0
git push origin api_sdk/go/v0.1.0
```

Consumers install a specific version with:

```bash
go get github.com/PaddlePaddle/PaddleOCR/api_sdk/go@v0.1.0
```

If the Go SDK reaches v2 or later, Go module rules require the module path to add
the major suffix, for example:

```go
module github.com/PaddlePaddle/PaddleOCR/api_sdk/go/v2
```

and tags should use the same subdirectory prefix, such as `api_sdk/go/v2.0.0`.

## Changelog

Maintain [`CHANGELOG.md`](CHANGELOG.md) for release notes that affect any SDK.
Language-specific notes may be grouped under TypeScript, Go, and Python headings
inside the same release entry.
