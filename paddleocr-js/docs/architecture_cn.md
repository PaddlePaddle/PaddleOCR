# 架构说明

[English](architecture.md) | 简体中文

## 项目结构

`paddleocr-js` 目录下主要有两部分：

- `packages/core`：浏览器 PaddleOCR SDK，发布到 npm 时包名为 `@paddleocr/paddleocr-js`
- `apps/demo`：依赖该 SDK 的 PP-OCR 演示应用

## SDK 包布局（`packages/core`）

```
src/
├── runtime/       — 推理运行时初始化
├── resources/     — 模型与资源管理
├── models/        — 模型接线
├── platform/      — 浏览器 / Worker 输入适配
├── worker/        — Worker 传输层
├── pipelines/     — 产线实现
├── viz/           — 可视化（可选）
├── types/         — 外部库类型声明
└── utils/         — 共享工具
```

当前高层产线入口为 `PaddleOCR.create()`，它负责协调：

1. 运行时初始化
2. 执行后端选择
3. 模型下载
4. 推理会话创建
5. OCR 产线执行

## Worker 执行模型

`PaddleOCR.create()` 支持两种执行模式：

- 主线程模式：返回 `PaddleOCR`，直接在调用线程上执行 OCR
- Worker 模式：返回 `WorkerBackedPaddleOCR`，将 OCR 生命周期调用转发到独立 Worker

Worker 模式下的运行流程如下：

1. `PaddleOCR.create({ worker: true })` 解析 OCR 选项并创建 `WorkerBackedPaddleOCR`
2. `WorkerBackedPaddleOCR` 通过 `WorkerTransportClient` 发送 `init` / `predict` / `dispose` 请求
3. OCR 产线层持有默认 Worker 工厂，并将其指向 `src/pipelines/ocr/worker-entry.ts`
4. `src/pipelines/ocr/worker-entry.ts` 将 `src/worker/entry.ts` 中的通用 Worker 引导逻辑与 OCR 专用处理逻辑绑定
5. `OcrPipelineRunner` 在 Worker 内运行 OpenCV.js、ONNX Runtime Web、模型加载、检测与识别
6. 结果和错误会被序列化后传回主线程

输入处理按环境拆分：

- 主线程：将浏览器输入标准化为可传输的负载
- Worker：将负载还原为 `cv.Mat` 等运行时输入

Worker 模式使用包内 Worker 路径，并在内部显式关闭 ONNX Runtime Web 的 wasm proxy。这样可以避免双层 Worker 叠加，并让包本身负责并发模型。

ONNX Runtime Web 在运行时需要 WASM 二进制。`ortOptions.wasmPaths` 对两种执行模式统一生效，设置一次即可控制主线程和 Worker 两侧的 WASM 加载位置：

```ts
PaddleOCR.create({
  ortOptions: { wasmPaths: "/assets/" }
});
```

设置了 `wasmPaths` 时，两种模式都会从指定路径拉取 WASM。未设置时，两种模式的回退行为不同：

- 主线程模式：ORT 通过使用方的打包工具解析 WASM（通常由打包工具把 `node_modules/onnxruntime-web/dist/` 下的 `.wasm` 文件拷贝到构建产物并自动改写 URL）
- Worker 模式：SDK 会回退到与构建时安装的 ORT 版本绑定的 CDN，并在控制台提示建议显式设置 `ortOptions.wasmPaths`

因此，在 Worker 模式下建议显式设置 `ortOptions.wasmPaths`，以保证两种模式使用同一套 WASM 版本。

## 应用侧职责

SDK 负责 OCR 运行时初始化与推理编排；宿主应用仍需负责：

- 运行环境所需的部署响应头
- 静态资源托管与模型 URL 配置
- `worker: true` 场景下支持产出并加载 Worker 的打包工具或运行时
- 应用界面、状态提示与可视化

这里的 `apps/` 目录就承载了这类宿主应用。
