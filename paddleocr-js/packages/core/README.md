# PaddleOCR.js SDK

English | [简体中文](README_cn.md)

`@paddleocr/paddleocr-js` is the browser SDK package for running PaddleOCR pipelines in the frontend.

## Install

```bash
npm install @paddleocr/paddleocr-js
```

## Quick Start

```js
import { PaddleOCR } from "@paddleocr/paddleocr-js";

const ocr = await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5",
  runtime: {
    backend: "auto"
  }
});

const result = await ocr.predict(fileOrBlob);
console.log(result.items);
```

## Construction Options

There are two main construction styles:

### 1. Direct parameters

You can directly specify model selection parameters in `PaddleOCR.create()`.

Use `lang + ocrVersion`:

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5"
});
```

Or use explicit model names:

```js
await PaddleOCR.create({
  textDetectionModelName: "PP-OCRv5_mobile_det",
  textRecognitionModelName: "PP-OCRv5_mobile_rec"
});
```

For browser usage, custom model files are also passed through direct parameters using asset descriptors:

```js
await PaddleOCR.create({
  textDetectionModelName: "my_det_model",
  textDetectionModelAsset: {
    id: "my-det-model",
    version: "2026-03-18",
    kind: "tar",
    url: "https://example.com/models/my_det_model.tar",
    entries: {
      model: "inference.onnx",
      config: "inference.yml"
    }
  },
  textRecognitionModelName: "my_rec_model",
  textRecognitionModelAsset: {
    id: "my-rec-model",
    version: "2026-03-18",
    kind: "tar",
    url: "https://example.com/models/my_rec_model.tar",
    entries: {
      model: "inference.onnx",
      config: "inference.yml"
    }
  }
});
```

### 2. Pipeline config

```js
import { PaddleOCR } from "@paddleocr/paddleocr-js";

const pipelineConfig = `
pipeline_name: OCR
SubModules:
  TextDetection:
    model_name: PP-OCRv5_mobile_det
  TextRecognition:
    model_name: PP-OCRv5_mobile_rec
`;

const ocr = await PaddleOCR.fromPipelineConfig(pipelineConfig);
```

`pipelineConfig` can be either YAML text or a parsed object.

If direct parameters and `pipelineConfig` are both provided, direct parameters take precedence.

All OCR model `inference.yml` files must define `model_name`, and PaddleOCR.js validates that value against the selected model name during initialization.

## Prediction Params

`ocr.predict(image, params?)` accepts both camelCase names and PaddleOCR-style snake_case names:

- `textDetLimitSideLen` or `text_det_limit_side_len`
- `textDetLimitType` or `text_det_limit_type`
- `textDetMaxSideLimit` or `text_det_max_side_limit`
- `textDetThresh` or `text_det_thresh`
- `textDetBoxThresh` or `text_det_box_thresh`
- `textDetUnclipRatio` or `text_det_unclip_ratio`
- `textRecScoreThresh` or `text_rec_score_thresh`

Supported `image` inputs include `Blob`, `ImageBitmap`, `ImageData`, `HTMLCanvasElement`, `HTMLImageElement`, and `cv.Mat`.

In worker mode (see next section), `cv.Mat` is not transferable and is therefore not supported as a worker input.

## Worker Mode

You can run the OCR pipeline inside a dedicated Worker while keeping the same high-level API:

```js
import { PaddleOCR } from "@paddleocr/paddleocr-js";

const ocr = await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5",
  worker: true,
  runtime: {
    backend: "wasm",
    wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/",
    numThreads: 2,
    simd: true
  }
});
```

Worker behavior:

- Worker mode uses the package worker path, not ONNX Runtime Web `env.wasm.proxy`.
- When `worker: true` is enabled, the package forces ORT wasm proxy off internally.
- Browser inputs are normalized on the main thread and transferred into the worker before inference runs.
- `cv.Mat` is only supported in the direct main-thread pipeline path.

## API

- `PaddleOCR.create(options)`
- `PaddleOCR.fromPipelineConfig(config, options?)`
- `ocr.initialize()`
- `ocr.getInitializationSummary()`
- `ocr.predict(image, params?)`
- `ocr.dispose()`
- `parseOcrPipelineConfigText(text)`
- `normalizeOcrPipelineConfig(config)`

## Package Layout

- `src/runtime`: runtime initialization and execution backend setup
- `src/resources`: model asset registry, download/cache, tar parsing, and asset resolution
- `src/models`: reusable model modules
- `src/platform`: browser and worker source adaptation helpers
- `src/worker`: worker transport client, protocol, and generic worker bootstrap
- `src/pipelines/ocr`: OCR config parsing, main-thread/worker-backed pipeline API assembly, default worker entry wiring, and shared execution runner

## Runtime Responsibilities

The SDK manages OpenCV.js and ONNX Runtime internally. The host application is still responsible for runtime environment concerns, including:

- COOP/COEP headers when enabling threaded WASM or WebGPU
- ONNX Runtime Web environment options such as wasm asset hosting paths, thread counts, and SIMD flags
- a bundler/runtime setup that can emit and load module workers when `worker: true` is used

## Browser Cache

Asset files are cached in the browser when Cache Storage is available, with an in-memory fallback for environments that do not expose the Cache API.
