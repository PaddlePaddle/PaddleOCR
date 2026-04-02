# PaddleOCR.js

English | [简体中文](README_cn.md)

Official browser OCR SDK and demo for PaddleOCR.

## Project structure

| Path | Role |
|------|------|
| `packages/core/` | Browser SDK sources; published to npm as **`@paddleocr/paddleocr-js`** |
| `apps/demo/` | Vite demo app that depends on the SDK |

## Quick start

```bash
npm install
npm run dev:demo
```

Other useful commands (still from `paddleocr-js/`):

```bash
npm run build
npm run test
npm run typecheck
npm run check
```

Workspace-scoped examples:

```bash
npm run build --workspace packages/core
npm run build --workspace apps/demo
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/development.md`](docs/development.md)
- [`docs/monorepo.md`](docs/monorepo.md)
- SDK-focused reference: [`packages/core/README.md`](packages/core/README.md)

## Acknowledgements

- [ONNX Runtime Web](https://onnxruntime.ai/) for browser-side ONNX inference
- [OpenCV.js](https://opencv.org/) for browser-side image processing
