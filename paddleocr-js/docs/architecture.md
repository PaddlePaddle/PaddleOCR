# Architecture

English | [简体中文](architecture_cn.md)

## Project structure

The `paddleocr-js` folder has two main parts:

- `packages/core`: the browser PaddleOCR SDK (published on npm as `@paddleocr/paddleocr-js`)
- `apps/demo`: a demo application for PP-OCR that consumes the SDK

## SDK package layout (`packages/core`)

```
src/
├── runtime/       — inference runtime setup
├── resources/     — model & asset management
├── models/        — model wiring
├── platform/      — browser/worker input adaptation
├── worker/        — worker transport layer
├── pipelines/     — pipeline implementations
├── viz/           — visualization (optional)
├── types/         — external type declarations
└── utils/         — shared utilities
```

The current high-level pipeline entry point is `PaddleOCR.create()`. It coordinates:

1. runtime initialization
2. execution backend selection
3. model download
4. inference session creation
5. OCR pipeline execution

## Worker execution model

`PaddleOCR.create()` supports 2 execution modes:

- main-thread mode: returns `PaddleOCR`, which runs OCR directly on the calling thread
- worker-backed mode: returns `WorkerBackedPaddleOCR`, which forwards OCR lifecycle calls to a dedicated worker

The runtime flow for worker mode is:

1. `PaddleOCR.create({ worker: true })` resolves OCR options and creates a `WorkerBackedPaddleOCR`
2. `WorkerBackedPaddleOCR` sends `init/predict/dispose` requests through `WorkerTransportClient`
3. the OCR pipeline layer owns the default worker factory and points it at `src/pipelines/ocr/worker-entry.ts`
4. `src/pipelines/ocr/worker-entry.ts` binds the generic worker bootstrap in `src/worker/entry.ts` to the OCR-specific worker handler
5. `OcrPipelineRunner` runs OpenCV.js, ONNX Runtime Web, model loading, detection, and recognition inside the worker
6. results and errors are serialized back to the main thread

Input handling is split by environment:

- main thread: browser inputs are normalized into transferable payloads
- worker: payloads are reconstructed into runtime inputs such as `cv.Mat`

Worker mode uses the package worker path and explicitly disables ONNX Runtime Web wasm proxy internally. This avoids stacking two worker layers and keeps the package responsible for the concurrency model.

ONNX Runtime Web requires WASM binaries at runtime. `ortOptions.wasmPaths` is a unified configuration that applies to both execution modes — setting it once controls where WASM is loaded in both main-thread and worker contexts:

```ts
PaddleOCR.create({
  ortOptions: { wasmPaths: "/assets/" }
});
```

When `wasmPaths` is set, both modes fetch WASM from the specified path. When it is not set, each mode falls back differently:

- main-thread mode: ORT resolves WASM through the consumer's bundler (the bundler copies `.wasm` files from `node_modules/onnxruntime-web/dist/` into the build output and rewrites the URLs automatically)
- worker mode: the SDK falls back to a CDN URL pinned to the ORT version installed at SDK build time, and emits a console warning recommending the consumer set `ortOptions.wasmPaths`

Setting `ortOptions.wasmPaths` explicitly is recommended for worker mode to ensure version consistency between the two modes.

## Application responsibilities

The SDK owns OCR runtime setup and inference orchestration. The host application still owns:

- deployment headers required by the runtime environment
- static asset hosting and model URL configuration
- worker-capable bundler/runtime support when `worker: true` is used
- application UI, status messaging, and visualization

Here, the `apps/` directory contains such host applications.
