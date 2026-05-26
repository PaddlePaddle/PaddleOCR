---
comments: true
---

# PaddleOCR 官方 API Python SDK

Python SDK 通过 `paddleocr` 包中的 `PaddleOCRClient` 和 `AsyncPaddleOCRClient` 调用 PaddleOCR 官方 API。它提交 OCR 或文档解析任务到官方托管服务，不运行本地推理，也不加载本地模型。

## 安装与认证

开发环境中可从当前源码安装 PaddleOCR：

```bash
python -m pip install -e .
export PADDLEOCR_ACCESS_TOKEN="your-api-token"
```

`PaddleOCRClient()` 默认读取 `PADDLEOCR_ACCESS_TOKEN`，也支持 `PaddleOCRClient(token="...")`。未提供 token 时会抛出 `AuthError`。

## 快速开始

```python
from paddleocr import PaddleOCRClient, Model

client = PaddleOCRClient()
result = client.ocr(
    file_url="https://example.com/invoice.pdf",
    model=Model.PP_OCRV5,
)
print(result.job_id, len(result.pages))
client.close()
```

传入本地文件时使用 `file_path`。`file_url` 与 `file_path` 必须二选一。

## 公共 API

Python SDK 的最终公共方法包括：

- `ocr(...)`：提交 OCR 任务，等待完成并返回 OCR 结果。
- `parse_document(...)`：提交文档解析任务，等待完成并返回文档解析结果。
- `submit_ocr(...)`：只提交 OCR 任务，返回任务对象。
- `submit_document_parsing(...)`：只提交文档解析任务，返回任务对象。
- `get_status(job_id)`：执行一次非阻塞状态查询，不等待完成。
- `wait_ocr_result(job)`：等待 OCR 任务完成并解析结果。
- `wait_document_parsing_result(job)`：等待文档解析任务完成并解析结果。

异步客户端 `AsyncPaddleOCRClient` 暴露任务操作的异步版本：`ocr`、`parse_document`、`submit_ocr`、`submit_document_parsing`、`get_status`、`get_batch_status`、`wait_ocr_result`、`wait_document_parsing_result` 和 `close`。

## 超时

```python
client = PaddleOCRClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` 限制一次 HTTP 请求，包括提交、查询状态和下载结果资源。`poll_timeout` 限制 `ocr`、`parse_document`、`wait_ocr_result` 和 `wait_document_parsing_result` 的总等待时间。

## 模型扩展

OCR 的 `model` 可省略，默认是 `Model.PP_OCRV5`。当前 PaddleOCR 官方 API 版本只开放 PP-OCRv5 作为 OCR 模型。文档解析的 `model` 可省略，默认是 `Model.PADDLE_OCR_VL_16`。SDK 通过 `is_ocr_model` 和 `is_document_parsing_model` 集中校验模型类型，后续新增模型时可在模型分类处集中扩展。

## 错误与资源保存

所有 SDK 错误都继承自 `PaddleOCRAPIError`，常见类型包括 `AuthError`、`InvalidRequestError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError`。

## 批量任务查询

提交任务时可传入 `batch_id`。之后可使用 `client.get_batch_status("batch-id")` 查询该批次下各任务的状态、进度和结果 URL。

## 文档解析参数类型

`PP-StructureV3` 使用 `PPStructureV3Options`，`PaddleOCR-VL`、`PaddleOCR-VL-1.5` 和 `PaddleOCR-VL-1.6` 使用 `PaddleOCRVLOptions`。这样可以避免把 VL 专属参数（如 `prompt_label`、`temperature`、`top_p`、`min_pixels`、`restructure_pages`）误传给 PP-StructureV3。
