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
| [`CONTRACT.md`](CONTRACT.md) | Cross-language public API contract for the first release. |
| [`VERSIONING.md`](VERSIONING.md) | SemVer, npm package, and Go module tag policy. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes for SDK-visible changes. |
| [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) | Release-readiness commands and manual gates. |
| [`typescript/README.md`](typescript/README.md) | Package-level README for the TypeScript SDK. |
| [`go/README.md`](go/README.md) | Package-level README for the Go SDK. |

Python SDK source is part of the main `paddleocr` package at
[`../paddleocr/_api_client`](../paddleocr/_api_client), so it does not have a
separate package README in this directory.

## Package Locations

| Language | Source location | User docs |
| --- | --- | --- |
| Python | [`../paddleocr/_api_client`](../paddleocr/_api_client) | [Python SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.md) |
| TypeScript | [`typescript`](typescript) | [TypeScript SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/typescript.md) |
| Go | [`go`](go) | [Go SDK](../docs/version3.x/inference_deployment/serving/paddleocr_official_api/go.md) |

All SDKs read `PADDLEOCR_ACCESS_TOKEN` by default. OCR defaults to PP-OCRv5;
document parsing defaults to PaddleOCR-VL-1.6. Keep detailed usage examples in
the user documentation, not in this directory.

## Validation

Run these from this directory unless a subdirectory is shown:

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

Run the extended checks in [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) before
public release.
