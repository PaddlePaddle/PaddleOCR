# Release

## Versioning

The `paddleocr-js` subproject is prepared for a Changesets-based release flow.

The **published npm package** is **`paddleocr-js`**, developed under [`packages/core/`](https://github.com/PaddlePaddle/PaddleOCR/tree/main/paddleocr-js/packages/core).

## Typical release flow

1. Merge changes into the default branch of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
2. Add or update changesets under `paddleocr-js/.changeset/` that describe user-facing changes.
3. Let the release workflow open or update the versioning pull request.
4. Merge the release pull request.
5. Publish to npm from the automated release workflow.

## Release requirements

- `NPM_TOKEN` configured in the repository secrets
- GitHub Actions enabled for the PaddleOCR repository
- package metadata validated before publishing

Source tree for this package: [github.com/PaddlePaddle/PaddleOCR/tree/main/paddleocr-js](https://github.com/PaddlePaddle/PaddleOCR/tree/main/paddleocr-js).
