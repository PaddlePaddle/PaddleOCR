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
  runtime: {
    backend: "auto"
  }
});

const result = await ocr.predict(fileOrBlob);
console.log(result.items);
```

## 构造方式

主要有两种构造方式：

### 1. 直接参数

你可以在 `PaddleOCR.create()` 中直接指定模型选择参数。

使用 `lang + ocrVersion`：

```js
await PaddleOCR.create({
  lang: "ch",
  ocrVersion: "PP-OCRv5"
});
```

或者显式指定模型名：

```js
await PaddleOCR.create({
  textDetectionModelName: "PP-OCRv5_mobile_det",
  textRecognitionModelName: "PP-OCRv5_mobile_rec"
});
```

在浏览器场景下，也可以通过资源描述对象，以直接参数方式传入自定义模型文件：

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

### 2. 产线配置

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

`pipelineConfig` 可以是 YAML 文本，也可以是解析后的对象。

如果同时提供直接参数和 `pipelineConfig`，则以直接参数为准。

所有 OCR 模型的 `inference.yml` 都必须定义 `model_name`，PaddleOCR.js 会在初始化阶段校验该值是否与所选模型名一致。

## 预测参数

`ocr.predict(image, params?)` 同时接受 camelCase 命名和 PaddleOCR 风格的 snake_case 命名：

- `textDetLimitSideLen` 或 `text_det_limit_side_len`
- `textDetLimitType` 或 `text_det_limit_type`
- `textDetMaxSideLimit` 或 `text_det_max_side_limit`
- `textDetThresh` 或 `text_det_thresh`
- `textDetBoxThresh` 或 `text_det_box_thresh`
- `textDetUnclipRatio` 或 `text_det_unclip_ratio`
- `textRecScoreThresh` 或 `text_rec_score_thresh`

支持的 `image` 输入包括 `Blob`、`ImageBitmap`、`ImageData`、`HTMLCanvasElement`、`HTMLImageElement` 和 `cv.Mat`。

在 worker 模式下（见下一节），`cv.Mat` 无法传输，因此不能作为 worker 输入。

## Worker 模式

你可以在专用 Worker 中运行 OCR 产线，同时保持相同的高层 API：

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
await viz.loadFont();

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

viz 模块会渲染一张左右对比的合成图像：左侧为带有检测框叠加的原始图像，右侧为识别出的文字。支持加载自定义字体以正确渲染中日韩等文字。

`deterministicColor(index)` 同样从 viz 子路径导出。它根据数字索引生成稳定的 RGB 颜色，内部用作检测框和文字标签的默认配色函数。当你构建自定义可视化并需要与内置渲染器保持一致的配色时，可以直接调用该函数。

## API

- `PaddleOCR.create(options)`
- `PaddleOCR.fromPipelineConfig(config, options?)`
- `ocr.initialize()`
- `ocr.getInitializationSummary()`
- `ocr.predict(image, params?)`
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

## 浏览器缓存

当浏览器支持 Cache Storage 时，资源文件会缓存到浏览器中；对于未暴露 Cache API 的环境，则回退为内存缓存。
