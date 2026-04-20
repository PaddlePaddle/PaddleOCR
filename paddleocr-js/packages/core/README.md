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
  ortOptions: {
    backend: "auto"
  }
});

const [result] = await ocr.predict(fileOrBlob);
console.log(result.items);
```

`predict` resolves to an **array** of `OcrResult` (one per input image). A single `Blob` / `File` still produces a one-element array—use destructuring or `results[0]`.

## Construction Options

There are two main construction styles:

### 1. Direct parameters

With direct parameters, you can specify models and configure inference batch sizes, ORT options, and other runtime settings.

**Model selection** — `lang` + `ocrVersion`:

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5"
});
```

**Model selection** — explicit model names:

```js
await PaddleOCR.create({
  textDetectionModelName: "PP-OCRv5_mobile_det",
  textRecognitionModelName: "PP-OCRv5_mobile_rec"
});
```

**Custom models** — provide model names and asset URLs for detection and recognition:

```js
await PaddleOCR.create({
  textDetectionModelName: "my_det_model",
  textDetectionModelAsset: {
    url: "https://example.com/models/my_det_model.tar"
  },
  textRecognitionModelName: "my_rec_model",
  textRecognitionModelAsset: {
    url: "https://example.com/models/my_rec_model.tar"
  }
});
```

**Custom model archive format and validation:**

- The downloaded bytes must be an **uncompressed ustar `.tar`**. The SDK does **not** gunzip **`.tar.gz`**; gzip-compressed payloads will usually fail to parse.
- The tar must contain **`inference.onnx`** and **`inference.yml`** (optionally under a subdirectory; matched by basename).
- **`inference.yml`** must define **`model_name`**, and it must match `textDetectionModelName` / `textRecognitionModelName`. This is checked during initialization after load.

Failures surface as **`Error`** during initialization (HTTP errors, missing tar entries, empty resources, missing/mismatched `model_name`, incomplete model config, ONNX session errors)—not silent fallbacks.

**Batch sizes, ORT options, and other runtime settings:**

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5",
  textDetectionBatchSize: 2,
  textRecognitionBatchSize: 8,
  ortOptions: {
    backend: "wasm",
    wasmPaths: "/assets/"
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
    batch_size: 2
  TextRecognition:
    model_name: PP-OCRv5_mobile_rec
    batch_size: 6
`;

const ocr = await PaddleOCR.create({ pipelineConfig });
```

`pipelineConfig` can be either YAML text or a parsed object.

If direct parameters and `pipelineConfig` are both provided, direct parameters take precedence.

## Prediction

### Params

`ocr.predict(image | images[], params?)` accepts both camelCase names and PaddleOCR-style snake_case names:

- `textDetLimitSideLen` or `text_det_limit_side_len`
- `textDetLimitType` or `text_det_limit_type`
- `textDetMaxSideLimit` or `text_det_max_side_limit`
- `textDetThresh` or `text_det_thresh`
- `textDetBoxThresh` or `text_det_box_thresh`
- `textDetUnclipRatio` or `text_det_unclip_ratio`
- `textRecScoreThresh` or `text_rec_score_thresh`

Supported `image` inputs include `Blob`, `ImageBitmap`, `ImageData`, `HTMLCanvasElement`, `HTMLImageElement`, and `cv.Mat`. Pass an array of these to run detection and recognition on multiple images in one call.

In worker mode (see next section), `cv.Mat` is not transferable and is therefore not supported as a worker input.

### Return value

Resolves to `Promise<OcrResult[]>`. Each `OcrResult` contains:

- `image`: `{ width, height }` for that source
- `items`: recognized lines (`poly`, `text`, `score`)
- `metrics`: `detMs`, `recMs`, `totalMs`, `detectedBoxes`, `recognizedCount` — box and line counts are per image; `detMs`, `recMs`, and `totalMs` cover the **entire** `predict()` call (so they are identical on every element when you pass multiple images)
- `runtime`: requested backend and provider metadata

## Worker Mode

You can run the OCR pipeline inside a dedicated Worker while keeping the same high-level API:

```js
import { PaddleOCR } from "@paddleocr/paddleocr-js";

const ocr = await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5",
  worker: true,
  ortOptions: {
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

## Visualization

The optional `@paddleocr/paddleocr-js/viz` subpath provides visualization utilities for rendering OCR results as images.

```js
import { OcrVisualizer } from "@paddleocr/paddleocr-js/viz";

const viz = new OcrVisualizer({
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" }
});

const blob = await viz.toBlob(imageBitmap, result);

// Trigger browser download
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "ocr_result.png";
a.click();
URL.revokeObjectURL(url);

viz.dispose();
```

A one-shot convenience function is also available:

```js
import { renderOcrToBlob } from "@paddleocr/paddleocr-js/viz";

const blob = await renderOcrToBlob(imageBitmap, result, {
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" }
});
```

The viz module renders a side-by-side composite image: the original image with detection box overlays on the left, and recognized text on the right. Custom fonts can be loaded for CJK text rendering. Visualization requires a **single** `OcrResult` (for one image, take the first element of the array returned by `predict`, e.g. `const [result] = await ocr.predict(image)`).

`deterministicColor(index)` is also exported from the viz subpath. It maps a numeric index to a stable RGB color and is used internally as the default color function for detection boxes and text labels. You can call it directly when building custom visualizations that need colors consistent with the built-in renderer.

## API

- `PaddleOCR.create(options)`
- `ocr.initialize()`
- `ocr.getInitializationSummary()`
- `ocr.predict(image | images[], params?)` → `Promise<OcrResult[]>`
- `ocr.dispose()`
- `parseOcrPipelineConfigText(text)`
- `normalizeOcrPipelineConfig(config)`
- `OcrVisualizer` (from `@paddleocr/paddleocr-js/viz`)
- `renderOcrToBlob` (from `@paddleocr/paddleocr-js/viz`)
- `deterministicColor` (from `@paddleocr/paddleocr-js/viz`)

## Package Layout

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

## Runtime Responsibilities

The SDK manages OpenCV.js and ONNX Runtime internally. The host application is still responsible for runtime environment concerns, including:

- COOP/COEP headers when enabling threaded WASM or WebGPU
- ONNX Runtime Web environment options such as wasm asset hosting paths, thread counts, and SIMD flags
- a bundler/runtime setup that can emit and load module workers when `worker: true` is used
