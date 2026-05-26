# API SDK Release Checklist

This checklist is the release-readiness gate for the PaddleOCR official API SDKs.
Run commands from the worktree root unless a command changes directories.

## Contract And Documentation

- Confirm `api_sdk/CONTRACT.md` still matches the public APIs documented in
  `api_sdk/README.md`, `api_sdk/PYTHON.md`, `api_sdk/typescript/README.md`, and
  `api_sdk/go/README.md`.
- Confirm the docs describe these SDKs as official API clients, not local
  inference wrappers.
- Confirm docs and examples use `PADDLEOCR_ACCESS_TOKEN` and only final public names:
  `get_status`, `getStatus`, `GetStatus`, `save_resource`, `saveResource`,
  `SaveResource`, `SaveOCRResultResources`, and
  `SaveDocumentParsingResultResources`.
- Confirm request and poll timeout names are distinct in every language:
  `request_timeout` / `poll_timeout`, `requestTimeout` / `pollTimeout`, and
  `WithRequestTimeout` / `WithPollTimeout`.
- Confirm TypeScript result-object bulk `saveResource` support is documented and
  any remaining deferred items are documented.

## Python

```bash
python -m pytest tests/test_api_client -q
python -c "import paddleocr; print(paddleocr.__version__)"
```

The test suite should cover token fallback, input validation, URL and file
submission, `get_status`, typed wait methods, timeouts, error precedence, result
parsing, resource saving, and CLI registration.

## TypeScript

```bash
cd api_sdk/typescript
npm run lint
npm run build
npm test
npm audit --audit-level=moderate
```

Before release, ensure generated build output is not accidentally committed
unless package publishing requires it.

## Go

```bash
cd api_sdk/go
go test ./...
go vet ./...
go test -race ./...
```

`go test -race ./...` is optional for day-to-day remediation but should pass
before public release when the local platform supports it.

## Repository Hygiene

```bash
git diff --check
git status --short --ignored=matching -- api_sdk/typescript
```

Confirm no `node_modules`, `dist`, caches, or generated temporary files are
tracked unintentionally.

## Release Decision

Ship only when:

- The validation commands above pass or every exception has an owner and release
  note.
- Public examples compile or type-check against the packaged SDK surface.
- The benchmark evaluation in `BENCHMARK_EVALUATION.md` has been reviewed and
  any deferred items are accepted by the release owner.
