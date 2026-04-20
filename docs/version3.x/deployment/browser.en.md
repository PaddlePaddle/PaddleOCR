---
comments: true
---

# PaddleOCR.js (browser deployment)

PaddleOCR provides **PaddleOCR.js**, a browser OCR SDK for running the PP-OCR pipeline in the browser. You can embed text detection and recognition in web apps and run inference on the client.

The npm package is **`@paddleocr/paddleocr-js`**. Source and demo live under [`paddleocr-js`](https://github.com/PaddlePaddle/PaddleOCR/tree/main/paddleocr-js) on GitHub.

## Install

```bash
npm install @paddleocr/paddleocr-js
```

## Quick start

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

## Construction options

Two styles: **direct parameters** to `PaddleOCR.create({ ... })`, or a **`pipelineConfig`** object.

### 1. Direct parameters

With direct parameters, you can specify models and set batch sizes, ORT options, and other runtime settings.

**Model selection — `lang` + `ocrVersion`:**

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5"
});
```

**Model selection — built-in model names:**

```js
await PaddleOCR.create({
  textDetectionModelName: "PP-OCRv5_mobile_det",
  textRecognitionModelName: "PP-OCRv5_mobile_rec"
});
```

**Custom models** — provide a name and asset URL for each of detection and recognition:

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

#### Custom model archive format and validation

The SDK downloads `textDetectionModelAsset.url` / `textRecognitionModelAsset.url` over HTTP(S) and parses the body as a **plain ustar tar (uncompressed)** archive. Ensure that:

| Requirement | Details |
|-------------|---------|
| Archive format | The response body must be an **uncompressed `.tar`**. The implementation does **not** gunzip **`.tar.gz`**; if you pass a gzip-compressed tarball, parsing will typically fail and an error will be thrown. |
| Required files | The tar must contain **`inference.onnx`** and **`inference.yml`** (they may live in a subdirectory; entries are matched by basename). |
| `model_name` | **`inference.yml`** must define a **`model_name`** that matches the `textDetectionModelName` / `textRecognitionModelName` you pass to `create`. This is checked after load during initialization. |

If you need to convert Paddle models into the ONNX model files used here, see [Obtaining ONNX models](obtaining_onnx_models.en.md). The standard model files produced by that workflow can then be packaged as a `.tar` following the rules above for use with PaddleOCR.js.

If the archive or model files do not meet these rules, initialization typically fails with an **`Error`** that describes the problem, for example: non-2xx download, missing `inference.onnx` / `inference.yml` in the tar, empty resources, missing or mismatched `model_name`, incomplete model config, or ONNX load failure. There is no silent fallback.

All selected OCR models must satisfy the `model_name` rules above.

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

`pipelineConfig` can be YAML text or a parsed object. In the browser, submodule `model_dir` must be **`null` or an asset object** (e.g. `{ url: "..." }`), not a local filesystem path string. If you want to start from a pipeline configuration exported by PaddleOCR / PaddleX, see the "Exporting Pipeline Configuration Files" section in [PaddleOCR and PaddleX](../paddleocr_and_paddlex.en.md); the exported YAML can be used as the basis for `pipelineConfig`, and any `model_dir` entries should then be adapted to browser-side asset objects.

If both direct parameters and `pipelineConfig` are provided, **direct parameters take precedence**.

## Prediction

### Params

`ocr.predict(image | images[], params?)` accepts both camelCase and PaddleOCR-style snake_case:

- `textDetLimitSideLen` or `text_det_limit_side_len`
- `textDetLimitType` or `text_det_limit_type`
- `textDetMaxSideLimit` or `text_det_max_side_limit`
- `textDetThresh` or `text_det_thresh`
- `textDetBoxThresh` or `text_det_box_thresh`
- `textDetUnclipRatio` or `text_det_unclip_ratio`
- `textRecScoreThresh` or `text_rec_score_thresh`

Supported `image` inputs include `Blob`, `ImageBitmap`, `ImageData`, `HTMLCanvasElement`, `HTMLImageElement`, and `cv.Mat`. Pass an array to run on multiple images in one call.

In **worker mode**, `cv.Mat` is not transferable and is not supported as input.

### Return value

Resolves to `Promise<OcrResult[]>`. Each `OcrResult` contains:

- `image`: `{ width, height }` for that source  
- `items`: recognized lines (`poly`, `text`, `score`)  
- `metrics`: `detMs`, `recMs`, `totalMs`, `detectedBoxes`, `recognizedCount` — box and line counts are per image; `detMs`, `recMs`, and `totalMs` cover the **entire** `predict()` call (identical on every element when you pass multiple images)  
- `runtime`: requested backend and provider metadata  

## Worker mode

You can run the pipeline inside a dedicated Worker while keeping the same high-level API:

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

Summary:

- Worker mode uses the package worker entry, not ONNX Runtime Web `env.wasm.proxy`  
- When `worker: true`, the package forces ORT wasm proxy off to avoid nested workers  
- Browser inputs are normalized on the main thread, then transferred to the worker  
- `cv.Mat` is only supported on the main-thread pipeline path  


## Visualization

The optional **`@paddleocr/paddleocr-js/viz`** subpath renders OCR results to images.

```js
import { OcrVisualizer } from "@paddleocr/paddleocr-js/viz";

const viz = new OcrVisualizer({
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" }
});

const blob = await viz.toBlob(imageBitmap, result);
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "ocr_result.png";
a.click();
URL.revokeObjectURL(url);

viz.dispose();
```

`renderOcrToBlob` and `deterministicColor` are also exported. Visualization takes a **single** `OcrResult` (for one image, use the first element of the `predict` result array).

## API summary

- `PaddleOCR.create(options)`  
- `ocr.initialize()` / `ocr.getInitializationSummary()`  
- `ocr.predict(image | images[], params?)` → `Promise<OcrResult[]>`  
- `ocr.dispose()`  
- `parseOcrPipelineConfigText(text)` / `normalizeOcrPipelineConfig(config)`  
- `OcrVisualizer`, `renderOcrToBlob`, `deterministicColor` (from `@paddleocr/paddleocr-js/viz`)  

## Host application responsibilities

The SDK manages OpenCV.js and ONNX Runtime internally. You still handle:

- **COOP/COEP** (and related headers) when enabling threaded WASM or WebGPU  
- **ORT environment options** (e.g. `wasmPaths`, threads, SIMD)  
- A bundler/runtime that can emit and load **module workers** when `worker: true`  
