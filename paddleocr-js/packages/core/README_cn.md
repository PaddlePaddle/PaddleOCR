# PaddleOCR.js SDK

[English](README.md) | 简体中文

`@paddleocr/paddleocr-js` 是在前端运行 PaddleOCR 产线的浏览器 SDK 包。

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

## 构造方式

主要有两种构造方式：

### 1. 直接参数

可通过直接参数指定模型，也可配置推理 batch size、ORT 选项等运行参数。

**模型选择 — `lang` + `ocrVersion`：**

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5"
});
```

**模型选择 — 显式模型名：**

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

**自定义模型包格式与校验行为：**

- 资源需为 **未压缩的标准 tar**（`.tar`）。SDK 按字节解析 tar，**不对 gzip 压缩包解压**；若 URL 指向 `.tar.gz` 等，通常会解析失败并报错。
- tar 内必须包含 **`inference.onnx`** 与 **`inference.yml`**（可在子目录中，按文件名匹配）。
- `inference.yml` 必须能解析出 **`model_name`**，且须与 `textDetectionModelName` / `textRecognitionModelName` **一致**；在 `initialize` 加载模型后会校验。

不满足时通常在初始化阶段以 **`Error`** 失败（下载失败、tar 中缺少条目、空资源、`model_name` 缺失或不匹配、模型配置不完整、ONNX 会话创建失败等），不会静默忽略。

**batch size、ORT 选项等运行参数：**

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

`pipelineConfig` 可以是 YAML 文本，也可以是解析后的对象。

如果同时提供直接参数和 `pipelineConfig`，则以直接参数为准。

## 预测

### 参数

`ocr.predict(image | images[], params?)` 同时接受 camelCase 命名和 PaddleOCR 风格的 snake_case 命名：

- `textDetLimitSideLen` 或 `text_det_limit_side_len`
- `textDetLimitType` 或 `text_det_limit_type`
- `textDetMaxSideLimit` 或 `text_det_max_side_limit`
- `textDetThresh` 或 `text_det_thresh`
- `textDetBoxThresh` 或 `text_det_box_thresh`
- `textDetUnclipRatio` 或 `text_det_unclip_ratio`
- `textRecScoreThresh` 或 `text_rec_score_thresh`

支持的 `image` 输入包括 `Blob`、`ImageBitmap`、`ImageData`、`HTMLCanvasElement`、`HTMLImageElement` 和 `cv.Mat`。传入上述类型的数组可在一次调用中对多张图像做检测与识别。

在 worker 模式下（见下一节），`cv.Mat` 无法传输，因此不能作为 worker 输入。

### 返回值

返回 `Promise<OcrResult[]>`。每个 `OcrResult` 包含：

- `image`：该图源的尺寸 `{ width, height }`
- `items`：识别行（`poly`、`text`、`score`）
- `metrics`：`detMs`、`recMs`、`totalMs`、`detectedBoxes`、`recognizedCount` — 检测框与文本行数为**每张图**统计；`detMs`、`recMs`、`totalMs` 表示**整次** `predict()` 调用的耗时（传入多图时，数组中每一项上的这三个值相同）
- `runtime`：请求的后端与各阶段 Provider 等元数据

## Worker 模式

你可以在专用 Worker 中运行 OCR 产线，同时保持相同的高层 API：

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

Worker 模式的行为：

- Worker 模式使用包内的 worker 路径，而不是 ONNX Runtime Web 的 `env.wasm.proxy`
- 启用 `worker: true` 时，包内部会强制关闭 ORT 的 wasm proxy
- 浏览器输入会先在主线程标准化，再传入 worker 执行推理
- `cv.Mat` 仅支持直接在主线程产线路径中使用

## 可视化

可选的 `@paddleocr/paddleocr-js/viz` 子路径提供了将 OCR 结果渲染为图像的可视化工具。

```js
import { OcrVisualizer } from "@paddleocr/paddleocr-js/viz";

const viz = new OcrVisualizer({
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" }
});

const blob = await viz.toBlob(imageBitmap, result);

// 触发浏览器下载
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "ocr_result.png";
a.click();
URL.revokeObjectURL(url);

viz.dispose();
```

也提供了一次性便捷函数：

```js
import { renderOcrToBlob } from "@paddleocr/paddleocr-js/viz";

const blob = await renderOcrToBlob(imageBitmap, result, {
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" }
});
```

viz 模块会渲染一张左右对比的合成图像：左侧为带有检测框叠加的原始图像，右侧为识别出的文字。支持加载自定义字体以正确渲染中日韩等文字。可视化需传入**单个** `OcrResult`（单张图时取 `predict` 返回数组的首项，例如 `const [result] = await ocr.predict(image)`）。

`deterministicColor(index)` 同样从 viz 子路径导出。它根据数字索引生成稳定的 RGB 颜色，内部用作检测框和文字标签的默认配色函数。当你构建自定义可视化并需要与内置渲染器保持一致的配色时，可以直接调用该函数。

## API

- `PaddleOCR.create(options)`
- `ocr.initialize()`
- `ocr.getInitializationSummary()`
- `ocr.predict(image | images[], params?)` → `Promise<OcrResult[]>`
- `ocr.dispose()`
- `parseOcrPipelineConfigText(text)`
- `normalizeOcrPipelineConfig(config)`
- `OcrVisualizer`（来自 `@paddleocr/paddleocr-js/viz`）
- `renderOcrToBlob`（来自 `@paddleocr/paddleocr-js/viz`）
- `deterministicColor`（来自 `@paddleocr/paddleocr-js/viz`）

## 包结构

```
src/
├── runtime/       — 推理运行时初始化
├── resources/     — 模型与资源管理
├── models/        — 模型接线
├── platform/      — 浏览器/worker 输入适配
├── worker/        — worker 传输层
├── pipelines/     — 产线实现
├── viz/           — 可视化（可选）
├── types/         — 外部库类型声明
└── utils/         — 共享工具
```

## 运行时职责边界

SDK 内部负责管理 OpenCV.js 和 ONNX Runtime。宿主应用仍需负责运行时环境相关事项，包括：

- 启用多线程 WASM 或 WebGPU 时所需的 COOP/COEP 响应头
- ONNX Runtime Web 的环境选项，例如 wasm 资源托管路径、线程数和 SIMD 开关
- 当使用 `worker: true` 时，能够产出并加载 module worker 的构建工具或运行时配置
