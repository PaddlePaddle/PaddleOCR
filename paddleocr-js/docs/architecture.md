# Architecture

## Subproject structure

The `paddleocr-js` folder is an npm workspace with two main roles:

- `packages/core`: the browser PaddleOCR SDK (published on npm as `paddleocr-js`)
- `apps/demo`: a demo application for PP-OCR that consumes the SDK

## SDK package layout (`packages/core`)

Inside `packages/core`, the SDK is organized into shared layers plus pipeline-specific
implementations:

- `src/runtime`: runtime initialization and execution backend setup
- `src/resources`: model registry, browser cache, tar parsing, and asset resolution
- `src/models`: model wiring plus preprocessing/postprocessing helpers
- `src/platform`: browser and worker helpers for turning user-provided image sources into runtime inputs
- `src/worker`: worker transport client, protocol, and generic message bootstrap
- `src/pipelines/ocr`: OCR config parsing, main-thread/worker-backed pipeline assembly, shared execution runner, and OCR-specific worker entry wiring

The current high-level pipeline entry point is `PaddleOCR.create()`. It coordinates:

1. runtime initialization
2. execution backend selection
3. model download and cache lookup
4. inference session creation
5. OCR pipeline execution

## Worker execution model

`PaddleOCR.create()` supports 2 execution modes:

- main-thread mode: returns `PaddleOCR`, which runs OCR directly on the calling thread
- worker-backed mode: returns `WorkerBackedPaddleOCR`, which forwards OCR lifecycle calls to a dedicated worker

The runtime flow for worker mode is:

1. `PaddleOCR.create({ worker: true })` resolves OCR options and creates a `WorkerBackedPaddleOCR`
2. `WorkerBackedPaddleOCR` sends `init/predict/dispose` requests through `WorkerTransportClient`
3. the OCR pipeline layer owns the default worker factory and points it at `src/pipelines/ocr/worker-entry.js`
4. `src/pipelines/ocr/worker-entry.js` binds the generic worker bootstrap in `src/worker/entry.js` to the OCR-specific worker handler
5. `OcrPipelineRunner` runs OpenCV.js, ONNX Runtime Web, model loading, detection, and recognition inside the worker
6. results and errors are serialized back to the main thread

Input handling is split by environment:

- main thread: browser inputs are normalized into transferable payloads
- worker: payloads are reconstructed into runtime inputs such as `cv.Mat`

Worker mode uses the package worker path and explicitly disables ONNX Runtime Web wasm proxy internally. This avoids stacking two worker layers and keeps the package responsible for the concurrency model.

## Application responsibilities

The SDK owns OCR runtime setup and inference orchestration. The host application still owns:

- deployment headers required by the runtime environment
- static asset hosting and model URL configuration
- worker-capable bundler/runtime support when `worker: true` is used
- application UI, status messaging, and visualization

In this subproject, the `apps/` directory contains such host applications.
