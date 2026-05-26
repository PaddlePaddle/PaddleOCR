---
comments: true
---

# PaddleOCR 官方 API 总览

PaddleOCR 官方 API SDK 是面向 PaddleOCR 官方 API 的客户端封装。它们会把本地文件或文件 URL 提交到官方托管服务，轮询异步任务并解析结果；它们不在本机运行 PaddleOCR 推理，也不会加载本地模型。

目前提供 Python、TypeScript、Go SDK，以及集成在 PaddleOCR CLI 中的 `paddleocr api` 命令。Python SDK 的源码相邻参考位于 `api_sdk/PYTHON.md`，TypeScript 和 Go 参考分别位于 `api_sdk/typescript/README.md` 与 `api_sdk/go/README.md`。本文档站页面是面向用户的官方文档入口。

## 认证

所有 SDK 和 CLI 都默认读取 `PADDLEOCR_ACCESS_TOKEN`：

```bash
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

也可以在构造客户端或运行 CLI 时显式传入 token。缺少 token 或鉴权失败时，SDK 会返回对应的类型化认证错误。

## 选择语言

- Python：适合已经使用 `paddleocr` Python 包的项目，提供同步 `PaddleOCRClient` 与异步 `AsyncPaddleOCRClient`。
- TypeScript：适合 Node.js 18 及以上的服务端项目。
- Go：适合需要静态类型、上下文取消和二进制部署的服务端项目。
- CLI：适合脚本、调试和无代码快速验证。

## 模型与任务

OCR 模型参数可省略，默认使用 PP-OCRv5。当前 PaddleOCR 官方 API 发布版本只开放 PP-OCRv5 作为 OCR 模型；各语言 SDK 都通过集中模型分类辅助函数校验 OCR 与文档解析模型，因此未来新增 OCR 模型时可以在模型定义处集中扩展，而不需要改动提交、轮询和保存结果的主流程。

文档解析任务的 `model` 参数可选，各 SDK 默认使用 PaddleOCR-VL-1.6。支持的模型包括 PP-StructureV3、PaddleOCR-VL、PaddleOCR-VL-1.5 和 PaddleOCR-VL-1.6。

## 结果与资源

便捷调用会提交任务、等待完成、下载结果 JSONL 并解析为类型化结果对象。显式提交接口适合需要非阻塞状态查询、并发等待或自定义调度的场景。

TypeScript 与 Go SDK 提供 `saveResource` / `SaveResource` 及结果对象批量保存帮助函数，用于将结果中的图片、Markdown 资源或其他输出资源保存到本地。资源下载不会向结果资源 URL 发送 PaddleOCR 官方 API 鉴权头，默认也不会覆盖已有文件。

## 错误处理

SDK 在用户文档层面暴露类型化错误，覆盖鉴权失败、参数校验失败、HTTP 非 2xx、网络错误、单次请求超时、轮询超时、远端任务失败、响应格式异常和结果解析失败等常见场景。CLI 会将错误输出到标准错误并返回非零退出码。
