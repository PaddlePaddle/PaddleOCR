---
comments: true
---

# PaddleOCR 官方 API 总览

PaddleOCR 官方 API SDK 是面向 PaddleOCR 官方 API 的客户端封装。它们会把本地文件或文件 URL 提交到官方托管服务，轮询异步任务并解析结果；它们不在本机运行 PaddleOCR 推理，也不会加载本地模型。

目前提供 Python、TypeScript、Go SDK，以及集成在 PaddleOCR CLI 中的 `paddleocr api` 命令。

## 文档入口

- [Python SDK](python.md)：适合已经使用 `paddleocr` Python 包的项目，提供同步 `PaddleOCRClient` 与异步 `AsyncPaddleOCRClient`。
- [TypeScript SDK](typescript.md)：适合 Node.js 18 及以上的服务端项目。
- [Go SDK](go.md)：适合需要静态类型、上下文取消和二进制部署的服务端项目。
- [CLI](cli.md)：适合脚本、调试和无代码快速验证。
