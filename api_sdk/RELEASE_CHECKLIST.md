# API SDK Release Checklist

English | [简体中文](RELEASE_CHECKLIST_cn.md)

This checklist is the release-readiness gate for the PaddleOCR official API SDKs.
Run commands from this `api_sdk` directory unless a command changes directories.

## Documentation Gate

- Confirm [`CONTRACT.md`](CONTRACT.md) matches the public APIs documented under
  [`../docs/version3.x/inference_deployment/serving/paddleocr_official_api/`](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/).
- Confirm `api_sdk/` files remain maintainer or package-level references, not a
  second copy of the user docs.
- Confirm docs and examples use `PADDLEOCR_ACCESS_TOKEN`, PaddleOCR-VL-1.6 as
  the default document parsing model, and only final public API names.
- Confirm [`VERSIONING.md`](VERSIONING.md) and [`CHANGELOG.md`](CHANGELOG.md)
  reflect the release version, npm package name, and Go tag strategy.

## Validation Commands

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

Before release, ensure generated build output is not accidentally committed
unless package publishing requires it. `go test -race ./...` may be skipped for
day-to-day remediation but should pass before public release when supported by
the local platform.

## Release Decision

Ship only when:

- The validation commands above pass or every exception has an owner and release
  note.
- Public examples compile or type-check against the packaged SDK surface.
- No `node_modules`, `dist`, caches, or generated temporary files are tracked
  unintentionally.
- TypeScript package metadata points to `@paddleocr/api-sdk`; Go release tags
  use the `api_sdk/go/vX.Y.Z` submodule tag format.
- Any deferred items are accepted by the release owner and documented in the
  release note.
