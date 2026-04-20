---
comments: true
---

# PaddleOCR.js（浏览器端部署）

PaddleOCR 提供可在浏览器中运行 PP-OCR 产线的浏览器端 OCR SDK **PaddleOCR.js**。您可以将文字检测与识别能力嵌入 Web 应用，在客户端完成推理。

npm 上的包名为 **`@paddleocr/paddleocr-js`**。源码与演示应用位于 GitHub 仓库的 [`paddleocr-js`](https://github.com/PaddlePaddle/PaddleOCR/tree/main/paddleocr-js) 目录。

## 安装

```bash
npm install @paddleocr/paddleocr-js
```

## 快速开始

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

`predict` 返回 **`OcrResult` 组成的数组**（每张输入图像对应一项）。传入单个 `Blob` / `File` 时也会得到长度为 1 的数组，请使用解构或 `results[0]` 取值。

## 构造选项

主要有两种构造方式：向 `PaddleOCR.create({ ... })` 传入**直接参数**，或传入 **产线配置** `pipelineConfig`。

### 1. 直接参数

可通过直接参数指定模型，也可设置推理 batch size、ORT 选项等运行参数。

**按语言与版本选择模型** — `lang` + `ocrVersion`：

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5"
});
```

**按内置模型名选择**：

```js
await PaddleOCR.create({
  textDetectionModelName: "PP-OCRv5_mobile_det",
  textRecognitionModelName: "PP-OCRv5_mobile_rec"
});
```

**自定义模型** — 为检测 / 识别分别指定模型名与资源地址：

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

**batch size 与 ORT 选项等运行参数**：

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

#### 自定义模型包格式与校验行为

SDK 通过 HTTP(S) 下载 `textDetectionModelAsset.url` / `textRecognitionModelAsset.url` 指向的资源，并按 **标准 ustar tar（未压缩）** 解析。请保证：

| 要求 | 说明 |
|------|------|
| 归档格式 | 响应体需为 **未压缩的 `.tar`**。当前实现不对 **`.tar.gz` / gzip** 做解压；若误传压缩包，解析结果通常不正确，随后会报错。 |
| 必需文件 | tar 内必须包含 **`inference.onnx`** 与 **`inference.yml`**（可在子目录中；按文件名匹配）。 |
| `model_name` | `inference.yml` 中必须能解析出 **`model_name`**，且须与您在 `create` 中传入的 `textDetectionModelName` / `textRecognitionModelName` **完全一致**。初始化加载模型后会校验。 |

如需从 Paddle 模型转换得到这里使用的 ONNX 模型文件，可参考 [获取 ONNX 模型](obtaining_onnx_models.md)。按该文档转换得到的标准模型文件，可按上述要求打包为 `.tar` 后提供给 PaddleOCR.js 使用。

若格式不满足要求，通常会在初始化阶段以 **抛出带明确信息的 `Error`** 失败，例如：下载非 2xx、tar 中找不到 `inference.onnx` / `inference.yml`、资源为空、`model_name` 缺失或与所选名称不一致、模型配置不完整，或 ONNX 无法加载。不会在后台静默失败。

所有通过上述方式选用的 OCR 模型，其 `inference.yml` 都应满足上述 `model_name` 约定。

### 2. 产线配置

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

`pipelineConfig` 可以是 YAML 文本，也可以是已解析的对象。在浏览器中，子模块的 `model_dir` 仅支持 **`null` 或资源描述对象**（形如 `{ url: "..." }`），不支持本地路径字符串。如需基于 PaddleOCR / PaddleX 导出的产线配置作为起点，可参考 [PaddleOCR 与 PaddleX](../paddleocr_and_paddlex.md) 中“导出产线配置文件”一节；导出的 YAML 可作为 `pipelineConfig` 的基础，再按浏览器端要求将其中的 `model_dir` 调整为资源描述对象。

若同时提供直接参数与 `pipelineConfig`，**以直接参数为准**。

## 预测

### 参数

`ocr.predict(image | images[], params?)` 同时接受 camelCase 与 PaddleOCR 风格的 snake_case：

- `textDetLimitSideLen` / `text_det_limit_side_len`
- `textDetLimitType` / `text_det_limit_type`
- `textDetMaxSideLimit` / `text_det_max_side_limit`
- `textDetThresh` / `text_det_thresh`
- `textDetBoxThresh` / `text_det_box_thresh`
- `textDetUnclipRatio` / `text_det_unclip_ratio`
- `textRecScoreThresh` / `text_rec_score_thresh`

支持的 `image` 类型包括：`Blob`、`ImageBitmap`、`ImageData`、`HTMLCanvasElement`、`HTMLImageElement`、`cv.Mat`。可传入上述类型的数组以在一次调用中处理多图。

在 **Worker 模式**下，`cv.Mat` 无法传输，不能作为 Worker 路径的输入。

### 返回值

返回 `Promise<OcrResult[]>`。每个 `OcrResult` 包含：

- `image`：该图源尺寸 `{ width, height }`
- `items`：识别行（`poly`、`text`、`score`）
- `metrics`：`detMs`、`recMs`、`totalMs`、`detectedBoxes`、`recognizedCount` — 框数与行数为**每张图**统计；`detMs`、`recMs`、`totalMs` 为**整次** `predict()` 调用耗时（多图时每一项上这三个值相同）
- `runtime`：请求的后端与各阶段 Provider 等元数据

## Worker 模式

可在独立 Worker 中运行产线，同时保持相同的高层 API：

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

行为概要：

- Worker 模式使用包内 Worker 脚本路径，而非 ONNX Runtime Web 的 `env.wasm.proxy`
- 启用 `worker: true` 时，包内会关闭 ORT 的 wasm proxy，避免双层 Worker
- 浏览器输入先在主线程标准化，再传入 Worker 推理
- `cv.Mat` 仅支持主线程产线路径


## 可视化

可选子路径 **`@paddleocr/paddleocr-js/viz`** 提供将 OCR 结果渲染为图像的工具。

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

也提供一次性函数 `renderOcrToBlob` 与 `deterministicColor`（用于与内置检测框配色一致）。可视化需传入**单个** `OcrResult`（单张图时取 `predict` 返回数组的首项）。

## API 摘要

- `PaddleOCR.create(options)`
- `ocr.initialize()` / `ocr.getInitializationSummary()`
- `ocr.predict(image | images[], params?)` → `Promise<OcrResult[]>`
- `ocr.dispose()`
- `parseOcrPipelineConfigText(text)` / `normalizeOcrPipelineConfig(config)`
- `OcrVisualizer`、`renderOcrToBlob`、`deterministicColor`（`@paddleocr/paddleocr-js/viz`）

## 宿主环境职责

SDK 内部管理 OpenCV.js 与 ONNX Runtime。您仍需自行处理：

- 启用多线程 WASM 或 WebGPU 时所需的 **COOP/COEP** 等响应头
- **ORT 环境选项**（如 `wasmPaths`、线程数、SIMD）
- 使用 **`worker: true`** 时，构建工具需能产出并加载 **module worker**
