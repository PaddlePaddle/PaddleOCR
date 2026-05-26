---
comments: true
---

# PaddleOCR 官方 API Python SDK

Python SDK 通过 `paddleocr` 包中的 `APIClient` 和 `AsyncAPIClient` 调用 PaddleOCR 官方 API。它提交 OCR 或文档解析任务到官方托管服务，不运行本地推理，也不加载本地模型。

## 安装与认证

开发环境中可从当前源码安装 PaddleOCR：

```bash
python -m pip install -e .
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

`APIClient()` 默认读取 `PADDLEOCR_ACCESS_TOKEN`，也支持 `APIClient(token="...")`。未提供 token 时会抛出 `AuthError`。

## 快速开始

```python
from paddleocr import APIClient, Model

client = APIClient()
result = client.ocr(
    file_url="https://example.com/invoice.pdf",
    model=Model.PP_OCRV5,
)
print(result.job_id, len(result.pages))
client.close()
```

传入本地文件时使用 `file_path`。`file_url` 与 `file_path` 必须二选一。

文档解析示例：

```python
from paddleocr import APIClient, DocParsingOptions, Model

client = APIClient()
result = client.parse_document(
    model=Model.PADDLE_OCR_VL_15,
    file_path="./report.pdf",
    options=DocParsingOptions(use_chart_recognition=True),
)
print(result.job_id, len(result.pages))
for page in result.pages:
    print(page.markdown_text)
client.close()
```

## 公共 API

Python SDK 的最终公共方法包括：

- `ocr(...)`：提交 OCR 任务，等待完成并返回 OCR 结果。
- `parse_document(...)`：提交文档解析任务，等待完成并返回文档解析结果。
- `submit_ocr(...)`：只提交 OCR 任务，返回任务对象。
- `submit_document_parsing(...)`：只提交文档解析任务，返回任务对象。
- `get_status(job_id)`：执行一次非阻塞状态查询，不等待完成。
- `wait_ocr_result(job)`：等待 OCR 任务完成并解析结果。
- `wait_document_parsing_result(job)`：等待文档解析任务完成并解析结果。
- `save_resource(resource, destination, overwrite=False)`：保存单个资源 URL 或结果对象中的资源。

异步客户端 `AsyncAPIClient` 暴露任务操作的异步版本：`ocr`、`parse_document`、`submit_ocr`、`submit_document_parsing`、`get_status`、`wait_ocr_result`、`wait_document_parsing_result` 和 `close`。它当前不提供 `save_resource` 协程方法。

## 超时

```python
client = APIClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` 限制一次 HTTP 请求，包括提交、查询状态和下载结果资源。`poll_timeout` 限制 `ocr`、`parse_document`、`wait_ocr_result` 和 `wait_document_parsing_result` 的总等待时间。

## 模型扩展

OCR 的 `model` 可省略，默认是 `Model.PP_OCRV5`。当前 PaddleOCR 官方 API 版本只开放 PP-OCRv5 作为 OCR 模型。

文档解析的 `model` 可选，默认使用 `Model.PADDLE_OCR_VL_15`；CLI 中也可通过 `--model` 覆盖。支持的文档解析模型包括 `Model.PP_STRUCTURE_V3`、`Model.PADDLE_OCR_VL` 和 `Model.PADDLE_OCR_VL_15`。SDK 通过 `is_ocr_model` 与 `is_document_parsing_model` 集中校验模型类型，后续新增模型时可在模型分类处集中扩展。

## 错误与资源保存

所有 SDK 错误都继承自 `PaddleOCRAPIError`，常见类型包括 `AuthError`、`InvalidRequestError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError`。

资源保存目前由同步客户端 `APIClient.save_resource` 提供。异步用户可以在获得结果资源 URL 后使用 `APIClient.save_resource` 保存资源，或在需要端到端异步 I/O 时自行实现异步下载。请勿使用 `await async_client.save_resource(...)`，该方法当前不存在。

`APIClient.save_resource` 可以保存单个资源 URL，也可以保存 `OCRResult` 或 `DocParsingResult` 中引用的全部资源。默认不覆盖已有文件，并且不会把 PaddleOCR 官方 API 的鉴权头发送到结果资源 URL。

更多源码相邻参考见 [api_sdk/PYTHON.md](../../../api_sdk/PYTHON.md)。
