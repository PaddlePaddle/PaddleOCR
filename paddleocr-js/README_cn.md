# PaddleOCR.js

[English](README.md) | 简体中文

PaddleOCR 官方浏览器 OCR SDK 与演示应用。

## 项目结构

| 路径             | 作用                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `packages/core/` | 浏览器 SDK 源码；发布到 npm 时的包名为 **`@paddleocr/paddleocr-js`** |
| `apps/demo/`     | 依赖该 SDK 的 Vite 演示应用                                          |

## 本地开发与演示

```bash
npm install
npm run dev:demo
```

其他常用命令：

```bash
npm run build
npm run test
npm run typecheck
npm run check
```

按 workspace 作用域：

```bash
npm run build --workspace packages/core
npm run build --workspace apps/demo
```

## 文档

| 说明 | 链接 |
|------|------|
| 架构说明 | [architecture_cn.md](docs/architecture_cn.md) |
| 开发指南 | [development_cn.md](docs/development_cn.md) |
| Monorepo 约定 | [monorepo_cn.md](docs/monorepo_cn.md) |
| SDK 包 README | [packages/core/README_cn.md](packages/core/README_cn.md) |

## 致谢

- [ONNX Runtime Web](https://onnxruntime.ai/)：提供浏览器端 ONNX 推理能力
- [OpenCV.js](https://opencv.org/)：提供浏览器端图像处理能力
