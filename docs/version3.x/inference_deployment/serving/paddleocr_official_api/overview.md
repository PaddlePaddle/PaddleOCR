---
comments: true
---

# PaddleOCR 官方 API 总览

PaddleOCR 官方 API SDK 是面向 PaddleOCR 官方 API 的客户端封装。它们会把本地文件或文件 URL 提交到官方托管服务，轮询异步任务并解析结果；它们不在本机运行 PaddleOCR 推理，也不会加载本地模型。

目前提供 Python、TypeScript、Go SDK，以及集成在 PaddleOCR CLI 中的 `paddleocr api` 命令。

## 认证

请先在 [AI Studio Access Token 页面](https://aistudio.baidu.com/account/accessToken) 获取访问令牌。

所有 SDK 和 CLI 都默认读取 `PADDLEOCR_ACCESS_TOKEN`：

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

也可以在构造客户端或运行 CLI 时显式传入 token。缺少 token 或鉴权失败时，SDK 会返回对应的类型化认证错误。

## 文档入口

- [Python SDK](python.md)：适合已经使用 `paddleocr` Python 包的项目，提供同步 `PaddleOCRClient` 与异步 `AsyncPaddleOCRClient`。
- [TypeScript SDK](typescript.md)：适合 Node.js 18 及以上的服务端项目。
- [Go SDK](go.md)：适合需要静态类型、上下文取消和二进制部署的服务端项目。
- [CLI](cli.md)：适合脚本、调试和无代码快速验证。

## 模型选择

OCR 模型参数可省略，默认使用 PP-OCRv5。当前 PaddleOCR 官方 API 发布版本只开放 PP-OCRv5 作为 OCR 模型。

文档解析任务的 `model` 参数可选，各 SDK 默认使用 PaddleOCR-VL-1.6。支持的模型包括 PP-StructureV3、PaddleOCR-VL、PaddleOCR-VL-1.5 和 PaddleOCR-VL-1.6。

## 结果与资源

各 SDK 都提供便捷调用、显式提交、状态查询、等待结果和资源保存能力。CLI 可输出 JSON，也可将结果对象引用的资源保存到本地目录。错误处理、超时和语言特定用法请见对应文档。
