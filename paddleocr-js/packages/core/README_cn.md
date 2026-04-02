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

## API

- `PaddleOCR.create(options)`
- `PaddleOCR.fromPipelineConfig(config, options?)`
- `ocr.initialize()`
- `ocr.getInitializationSummary()`
- `ocr.predict(image, params?)`
- `ocr.dispose()`
- `parseOcrPipelineConfigText(text)`
- `normalizeOcrPipelineConfig(config)`

## 包结构

- `src/runtime`：运行时初始化与执行后端设置
- `src/resources`：模型资源注册表、下载与缓存、tar 解析、资源解析
- `src/models`：可复用模型模块
- `src/platform`：浏览器与 worker 场景下的源码适配辅助
- `src/worker`：worker 传输客户端、协议与通用 worker 启动代码
- `src/pipelines/ocr`：OCR 配置解析、主线程与 worker 双路径产线 API 组装、默认 worker 入口接线以及共享执行器

## 运行时职责边界

SDK 内部负责管理 OpenCV.js 和 ONNX Runtime。宿主应用仍需负责运行时环境相关事项，包括：

- 启用多线程 WASM 或 WebGPU 时所需的 COOP/COEP 响应头
- ONNX Runtime Web 的环境选项，例如 wasm 资源托管路径、线程数和 SIMD 开关
- 当使用 `worker: true` 时，能够产出并加载 module worker 的构建工具或运行时配置

## 浏览器缓存

当浏览器支持 Cache Storage 时，资源文件会缓存到浏览器中；对于未暴露 Cache API 的环境，则回退为内存缓存。
