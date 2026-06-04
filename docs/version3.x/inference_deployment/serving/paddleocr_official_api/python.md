---
comments: true
---

# PaddleOCR 官方 API Python SDK

Python SDK 通过 `paddleocr` 包中的 `PaddleOCRClient` 和 `AsyncPaddleOCRClient` 调用 PaddleOCR 官方 API。它提交 OCR 或文档解析任务到官方托管服务，不运行本地推理，也不加载本地模型。

## 安装与认证

先按 [安装 `paddleocr`](../../../installation.md#install-paddleocr) 安装 Python 包。安装 `paddleocr` 本体后即可使用此功能，无需安装额外依赖组。

请先在 [AI Studio Access Token 页面](https://aistudio.baidu.com/account/accessToken) 获取访问令牌。

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
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

Python SDK 常用公共方法包括：

- `ocr(...)`：提交 OCR 任务，等待完成并返回 OCR 结果。
- `parse_document(...)`：提交文档解析任务，等待完成并返回文档解析结果。
- `submit_ocr(...)`：只提交 OCR 任务，返回任务对象。
- `submit_document_parsing(...)`：只提交文档解析任务，返回任务对象。
- `get_status(job_id)`：执行一次非阻塞状态查询，不等待完成。
- `wait_ocr_result(job)`：等待 OCR 任务完成并解析结果。
- `wait_document_parsing_result(job)`：等待文档解析任务完成并解析结果。
- `save_resource(resource_url, destination)`：保存单个资源 URL。
- `save_ocr_result_resources(result, destination)`：保存 OCR 结果对象引用的资源。
- `save_document_parsing_result_resources(result, destination)`：保存文档解析结果对象引用的资源。

异步客户端 `AsyncPaddleOCRClient` 暴露上述任务操作和资源保存方法的异步版本。

## 模型选择

表中的 `Model` 枚举是官方 API 模型名字符串的类型安全写法，提交请求时会转换为对应的实际模型名。也可以直接传入官方 API 模型名字符串，例如 `model="PaddleOCR-VL-1.6"`。

| 任务 | 适用接口 | 默认模型 | 可选模型 | 参数类型 |
| --- | --- | --- | --- | --- |
| OCR | `ocr`、`submit_ocr`、`wait_ocr_result` | `Model.PP_OCRV5` | `Model.PP_OCRV5` | `OCROptions` |
| 文档解析 | `parse_document`、`submit_document_parsing`、`wait_document_parsing_result` | `Model.PADDLE_OCR_VL_16` | `Model.PP_STRUCTURE_V3`、`Model.PADDLE_OCR_VL`、`Model.PADDLE_OCR_VL_15`、`Model.PADDLE_OCR_VL_16` | 选择 `PP-StructureV3` 时传入 `PPStructureV3Options`；选择 `PaddleOCR-VL` 系列模型时传入 `PaddleOCRVLOptions`。 |

## 配置与参数

### 客户端配置

```python
client = PaddleOCRClient(
    request_timeout=300.0,
    poll_timeout=600.0,
)
```

`request_timeout` 限制一次 HTTP 请求，包括提交、查询状态和下载结果资源。`poll_timeout` 限制 `ocr`、`parse_document`、`wait_ocr_result` 和 `wait_document_parsing_result` 的总等待时间。

通过环境变量 `PADDLEOCR_BASE_URL` 或 `base_url` 参数指定自定义服务地址：

```python
client = PaddleOCRClient(base_url="https://my-proxy.com/paddle")
```

### 请求参数

SDK 的参数名使用 Python 惯用的 snake_case，提交请求时自动转换为官方 API 的 camelCase。只需传入非 `None` 的字段，未设置的字段使用服务端默认值。完整字段定义见各 Options 类型源码或官方 API 参考。

#### OCROptions（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `use_doc_orientation_classify` | bool | 文档方向分类 |
| `use_doc_unwarping` | bool | 文档扭曲矫正 |
| `visualize` | bool | 是否返回可视化结果图 |

#### PPStructureV3Options（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `use_table_recognition` | bool | 表格识别 |
| `use_formula_recognition` | bool | 公式识别 |
| `use_chart_recognition` | bool | 图表识别 |
| `prettify_markdown` | bool | Markdown 美化 |

#### PaddleOCRVLOptions（常用字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `use_layout_detection` | bool | 版面检测 |
| `use_chart_recognition` | bool | 图表识别 |
| `temperature` | float | 采样温度 |
| `prettify_markdown` | bool | Markdown 美化 |

## 错误处理

所有 SDK 错误都继承自 `PaddleOCRAPIError`，常见类型包括 `AuthError`、`InvalidRequestError`、`RateLimitError`、`ServiceUnavailableError`、`APIError`、`NetworkError`、`JobFailedError`、`RequestTimeoutError`、`PollTimeoutError`、`ResponseFormatError` 和 `ResultParseError`。

## 批量任务查询

提交任务时可传入 `batch_id`。之后可使用 `client.get_batch_status("batch-id")` 查询该批次下各任务的状态、进度和结果 URL。

## 官方 API 参考

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Kmfl2ycs0)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/Fmfz6oh2e)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/2mh4okm66)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Cmkz2m0ma)

## 配额与错误码

- [API 配额规则和错误码说明](https://ai.baidu.com/ai-doc/AISTUDIO/Xmjclapam)
