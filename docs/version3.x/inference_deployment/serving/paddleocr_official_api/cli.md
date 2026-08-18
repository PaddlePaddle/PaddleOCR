---
comments: true
---

# PaddleOCR 官方 API CLI

`paddleocr api` 是 PaddleOCR CLI 中调用 PaddleOCR 官方 API 的子命令。它把文件 URL 或本地文件提交到官方托管服务，等待任务完成并输出解析结果；它不运行本地推理。

## 安装与认证

先按 [安装 `paddleocr`](../../../installation.md#install-paddleocr) 安装 Python 包。安装 `paddleocr` 本体后即可使用此功能，无需安装额外依赖组。

请先在 [AI Studio Access Token 页面](https://aistudio.baidu.com/account/accessToken) 获取访问令牌。

CLI 默认读取 `PADDLEOCR_ACCESS_TOKEN`：

```bash
export PADDLEOCR_ACCESS_TOKEN="your-access-token"
```

也可以使用 `--token` 显式传入 token。

## 基本用法

```bash
paddleocr api \
  --model_type ocr \
  --file_url https://example.com/invoice.pdf
```

`--model_type` 必填，可选值为 `ocr` 或 `doc_parsing`。`--file_url` 与 `--file_path` 必须二选一。

## 常用参数

- `--model_type`：任务类型，`ocr` 或 `doc_parsing`。
- `--model`：模型名称。
- `--file_url`：待处理文件 URL。
- `--file_path`：待上传并处理的本地文件路径。
- `--base_url`：PaddleOCR API 服务的 base URL；缺省使用官方服务地址（也可通过 `PADDLEOCR_BASE_URL` 环境变量设置）。
- `--request_timeout`：一次 HTTP 请求的超时时间，单位为秒。
- `--poll_timeout`：等待远端任务完成的总超时时间，单位为秒。
- `--output`：输出 JSON 文件路径；省略时打印到标准输出。
- `--save_resources`：保存结果对象引用资源的目录。
- `--overwrite_resources`：保存资源时覆盖已有文件。
- `--page_ranges`：页码范围，例如 `2,4-6`。
- `--use_doc_orientation_classify`：文档方向分类，`True` 或 `False`。
- `--use_doc_unwarping`：文档扭曲矫正，`True` 或 `False`。
- `--use_textline_orientation`：文本行方向检测，`True` 或 `False`。
- `--text_det_limit_side_len`：文本检测图像边长限制。
- `--text_det_limit_type`：边长限制类型，`min` 或 `max`。
- `--text_rec_score_thresh`：文本识别置信度阈值。
- `--use_layout_detection`：版面检测，`True` 或 `False`。
- `--use_seal_recognition`：印章识别，`True` 或 `False`。
- `--use_table_recognition`：表格识别，`True` 或 `False`。
- `--use_formula_recognition`：公式识别，`True` 或 `False`。
- `--use_chart_recognition`：图表识别，`True` 或 `False`。
- `--visualize`：可视化结果图，`True` 或 `False`。
- `--prettify_markdown`：markdown 美化，`True` 或 `False`。

## OCR 示例

```bash
paddleocr api \
  --model_type ocr \
  --model PP-OCRv5 \
  --file_path ./invoice.pdf \
  --request_timeout 300 \
  --poll_timeout 600 \
  --output ocr-result.json
```

## 文档解析示例

```bash
paddleocr api \
  --model_type doc_parsing \
  --file_url https://example.com/report.pdf \
  --use_chart_recognition True \
  --save_resources ./doc-assets \
  --output doc-result.json
```

## 模型选择

| 任务 | `--model_type` | 默认模型 | 可选模型 |
| --- | --- | --- | --- |
| OCR | `ocr` | `PP-OCRv6` | `PP-OCRv5`、`PP-OCRv5-latin`、`PP-OCRv6` |
| 文档解析 | `doc_parsing` | `PaddleOCR-VL-1.6` | `PP-StructureV3`、`PaddleOCR-VL`、`PaddleOCR-VL-1.5`、`PaddleOCR-VL-1.6` |

## 输出行为

命令成功时输出格式化 JSON。OCR 结果包含 `jobId` 和每页的 `prunedResult`、`ocrImageUrl`；文档解析结果包含 `jobId` 和每页的 `markdownText`、`markdownImages`、`outputImages`。如果指定 `--output`，CLI 写入该文件并打印保存位置；否则直接打印到标准输出。指定 `--save_resources` 时，CLI 会把结果对象引用的资源保存到目标目录。

错误会输出到标准错误并返回非零退出码。常见原因包括缺少 `PADDLEOCR_ACCESS_TOKEN`、模型与 `--model_type` 不匹配、请求超时、轮询超时、远端任务失败或响应格式异常。

## 官方 API 参考

- [PP-OCRv5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Kmfl2ycs0)
- [PP-StructureV3 API](https://ai.baidu.com/ai-doc/AISTUDIO/Fmfz6oh2e)
- [PaddleOCR-VL API](https://ai.baidu.com/ai-doc/AISTUDIO/2mh4okm66)
- [PaddleOCR-VL-1.5 API](https://ai.baidu.com/ai-doc/AISTUDIO/Cmkz2m0ma)

## 配额与错误码

- [API 配额规则和错误码说明](https://ai.baidu.com/ai-doc/AISTUDIO/Xmjclapam)
