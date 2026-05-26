# PaddleOCR 官方 API SDK 契约

[English](CONTRACT.md) | 简体中文

本文定义 PaddleOCR 官方 API SDK 首个公开版本的公共接口面。当前分支 API 尚未发布，因此实现团队应根据本契约重命名、重塑或移除已有草稿 API；未发布分支 API 不需要保持向后兼容。

这些 SDK 是 PaddleOCR 服务的官方 API 封装。它们不运行本地 PaddleOCR 推理，不加载本地 PaddleOCR 模型，也不提供离线 OCR 执行能力。

## 客户端选项

所有 SDK 必须以符合语言习惯的名称支持相同行为。

| 选项 | 环境变量 | Python | TypeScript | Go | 必须行为 |
| --- | --- | --- | --- | --- | --- |
| API token / `PADDLEOCR_ACCESS_TOKEN` | `PADDLEOCR_ACCESS_TOKEN` | `token` | `token` | `Token` | 用于认证 PaddleOCR 官方 API 请求。缺省 token 可从 `PADDLEOCR_ACCESS_TOKEN` 读取；若两者都不存在，认证客户端必须在构造阶段以认证错误失败，除非未来契约明确加入无认证模式。 |
| Base URL | 无 | `base_url` | `baseUrl` | `BaseURL` | 覆盖 PaddleOCR 官方 API 端点。默认值为 SDK 发布文档中的生产端点。SDK 必须一致地规范化尾部斜杠，避免拼接路径时产生双斜杠或缺少分隔符。 |
| 请求超时 | 无 | `request_timeout` | `requestTimeout` | `RequestTimeout` | 单次 HTTP 请求的最大耗时，包括提交、状态查询和结果下载请求。它与轮询超时相互独立。 |
| 轮询超时 | 无 | `poll_timeout` | `pollTimeout` | `PollTimeout` | 等待异步任务进入终态的总耗时上限。它与请求超时相互独立。 |

## 操作

各 SDK 必须暴露语义一致、但名称符合语言习惯的操作。

| 操作 | Python | TypeScript | Go | 行为 |
| --- | --- | --- | --- | --- |
| OCR 便捷调用 | `ocr(...)` | `ocr(...)` | `OCR(...)` | 提交 OCR 任务、等待完成、下载并解析 OCR 结果，返回 `OCRResult`。 |
| 文档解析便捷调用 | `parse_document(...)` | `parseDocument(...)` | `ParseDocument(...)` | 提交文档解析任务、等待完成、下载并解析文档解析结果，返回 `DocParsingResult`。 |
| 提交 OCR | `submit_ocr(...)` | `submitOcr(...)` | `SubmitOCR(...)` | 启动 OCR 任务并立即返回 `Job`，不等待完成。 |
| 提交文档解析 | `submit_document_parsing(...)` | `submitDocumentParsing(...)` | `SubmitDocumentParsing(...)` | 启动文档解析任务并立即返回 `Job`，不等待完成。 |
| 获取状态 | `get_status(job_id)` | `getStatus(jobId)` | `GetStatus(ctx, jobID)` | 只执行一次非阻塞状态请求，返回 `JobStatus`。 |
| 等待 OCR 结果 | `wait_ocr_result(job)` | `waitOcrResult(job)` | `WaitOCRResult(ctx, job)` | 轮询 OCR 任务直到完成，并返回 `OCRResult`。 |
| 等待文档解析结果 | `wait_document_parsing_result(job)` | `waitDocumentParsingResult(job)` | `WaitDocumentParsingResult(ctx, job)` | 轮询文档解析任务直到完成，并返回 `DocParsingResult`。 |
| 资源保存/下载 | 不暴露 | `saveResource(...)` | `SaveResource(...)` | 按 SDK 的结果 URL 处理规则和覆盖规则下载或保存结果资源。TypeScript 与 Go 还支持结果对象批量保存帮助函数。 |

## 数据模型

`Job` 表示已被服务接受的异步托管任务。它必须携带 `jobId` 以及足够的模型和任务信息，使 SDK 无需检查结果字段即可判断任务是 OCR 还是文档解析。

`JobStatus` 表示任务当前状态。它必须携带 `jobId`、`state`、`progress` 和终态失败时的错误信息字段。`progress` 应保留服务返回值，并在服务未返回时可为空或可选。

基线任务状态为 `pending`、`running`、`done` 和 `failed`。`done` 与 `failed` 是终态。未知状态应产生 `ResponseFormatError`，除非未来契约有意扩展。如果服务新增 `canceled` 或 `expired` 等状态，必须先更新本契约，再由 SDK 作为公共状态暴露。

`OCRResult` 与 `DocParsingResult` 必须按任务类型建模。它们可以包含符合语言习惯的便利结构，但公共形态必须保留用户消费 OCR 和文档解析输出所需的服务结果数据，避免要求用户自行解析原始 HTTP 响应。

`OCRResult` 至少应暴露 `jobId` 和 `pages`。每个 OCR 页面必须暴露剪枝结果或原始 OCR 载荷，例如 `prunedResult`、`raw` 或等价的语言习惯字段，并在服务返回时提供可选 OCR 图片 URL。

`DocParsingResult` 至少应暴露 `jobId` 和 `pages`。每个文档解析页面必须在服务返回时暴露 `markdownText` 以及 Markdown / 输出图片资源映射。

结果模型可以包含原始载荷逃生口供高级用户使用，但公共结果不能只提供 raw-only 形态。

语言支持时，任务状态必须类型化或枚举化。Python 应暴露类型化字符串字面量或枚举，TypeScript 应暴露字符串联合或枚举，Go 应暴露具名字符串类型及常量。

## 模型分类

OCR API 必须接受类型化模型参数，并默认使用 PP-OCRv5：

- Python：`ocr(..., model=Model.PP_OCRV5)` 与 `submit_ocr(..., model=Model.PP_OCRV5)`。
- TypeScript：`OCRRequest.model?: Model`，默认 `Model.PPOCRv5`。
- Go：`OCRRequest.Model Model`，零值时默认 `PPOCRv5`。

文档解析 API 必须接受类型化模型参数，并默认使用 PaddleOCR-VL-1.6：

- Python：`parse_document(..., model=Model.PADDLE_OCR_VL_16)` 与 `submit_document_parsing(..., model=Model.PADDLE_OCR_VL_16)`。
- TypeScript：`DocParsingRequest.model?: Model`，默认 `Model.PaddleOCRVL16`。
- Go：`DocParsingRequest.Model Model`，零值时默认 `PaddleOCRVL16`。

本发布版本仅支持 PP-OCRv5 作为 OCR 模型，但 SDK 提交和等待逻辑必须具备模型扩展性。各 SDK 必须集中实现模型分类帮助函数，例如 `is_ocr_model` / `is_document_parsing_model`、`isOCRModel` / `isDocumentParsingModel`、`IsOCRModel` / `IsDocumentParsingModel`。OCR 等待方法必须通过 OCR 分类帮助函数校验任务，而不是直接判断是否等于 PP-OCRv5。文档解析提交和等待校验必须通过同一分类层拒绝 OCR 模型。

## 资源保存

资源保存/下载帮助函数必须按语言习惯接受结果对象或资源 URL 以及目标路径。当结果对象包含多个可下载资源时，帮助函数可以保存所有符合条件的资源，或要求文档化的选择器。

帮助函数必须返回已保存文件路径、路径列表，或包含已保存目标和跳过资源的类型化摘要。返回值必须足以让调用者知道写入了什么。

帮助函数不得静默覆盖已有文件。只有调用者显式传入覆盖选项时才允许覆盖。

网络和下载失败必须按错误优先级映射到 SDK 的网络错误或 API/HTTP 错误。文件系统失败必须以语言习惯的文件系统错误或文档化 SDK 错误暴露。

## 命名规则

`get_status`、`getStatus` 和 `GetStatus` 专用于非阻塞状态 API。它们不得等待终态，也不得下载或解析结果载荷。

状态 API 不得命名为 `get_result`、`getResult` 或 `GetResult`。返回结果的 API 必须明确体现等待行为和任务类型，例如 `wait_ocr_result`、`waitOcrResult` 和 `WaitOCRResult`。

SDK 不得根据 JSONL 字段是否存在来推断结果类型。结果解析必须由 `Job`、`JobStatus`、等待方法或另一个文档化类型判别字段中的显式任务/模型信息选择。

请求超时和轮询超时是两个独立概念。请求超时作用于单次 HTTP 操作；轮询超时作用于异步完成等待循环的总耗时。

SDK 不得为了未发布分支兼容而保留旧草稿别名。特别是不要保留 `get_result`、`getResult`、`GetResult` 或单一 timeout 选项作为兼容别名。

## 错误分类

所有 SDK 必须为相同失败模式暴露类型化错误或文档化错误类/类别：

| 错误类别 | 含义 |
| --- | --- |
| Auth | 凭证缺失、无效或被拒绝，包括缺少 `token` 和 `PADDLEOCR_ACCESS_TOKEN`。公开错误名应为 `AuthError` 或语言习惯等价名称。 |
| Invalid request | 用户输入未通过 SDK 侧校验，或被服务判定为无效 API 参数。公开错误名应为 `InvalidRequestError` 或语言习惯等价名称。 |
| API/HTTP | PaddleOCR 官方 API 返回非成功 HTTP 状态或文档化 API 错误响应。 |
| Network | DNS、连接、TLS、socket 或其他在收到有效 HTTP 响应前发生的传输失败。 |
| Job failed | 已提交任务进入服务报告的 `failed` 终态。 |
| Request timeout | 单次 HTTP 请求超过配置的请求超时。 |
| Poll timeout | 等待操作在任务进入终态前超过配置的轮询超时。 |
| File not found | SDK 所需的本地输入路径或目标父目录不存在。公开错误名应为 `FileNotFoundError` 或语言习惯等价名称。 |
| Response format | 传输成功的响应缺少必需字段、包含未知状态值，或以其他方式违反文档化 API 响应 schema。公开错误名应为 `ResponseFormatError` 或语言习惯等价名称。 |
| Result parse | 结果资源获取成功后，结果载荷解析失败，包括 JSONL 格式错误。公开错误名应为 `ResultParseError` 或语言习惯等价名称。 |

错误优先级必须跨语言一致：

- 本地输入文件缺失和目标父目录缺失是 `FileNotFoundError` 或语言习惯等价错误，并优先于通用 SDK 侧 `InvalidRequestError`。
- 网络请求前的 SDK 侧语义校验错误是 `InvalidRequestError`，包括同时提供文件 URL 和文件路径、两者都未提供、不支持的模型、等待方法任务类型不匹配，以及本地校验的无效页码范围。
- 缺少认证客户端 token 时，客户端构造阶段以 `AuthError` 失败，除非未来增加显式无认证模式。
- HTTP 401 和 403 是 `AuthError`。
- 服务端 HTTP 400 是 `InvalidRequestError`，并在可用时保留服务消息。
- 其他非 2xx HTTP 响应是 API 错误。
- HTTP 2xx 响应体格式错误是 `ResponseFormatError`。
- JSONL 或结果载荷解析失败是 `ResultParseError`。
- 将 OCR 任务传给文档解析等待方法，或将文档解析任务传给 OCR 等待方法，是 `InvalidRequestError`。

## 跨语言行为规则

HTTP 2xx 只表示传输成功。SDK 返回公共数据模型前仍必须按预期 API schema 校验响应体。

成功传输但格式错误的响应是 `ResponseFormatError`。这包括缺少 `jobId`、缺少状态、已完成任务缺少结果 URL，或其他必需字段违规。

JSONL 格式错误是 `ResultParseError`。一旦结果资源已成功获取，JSONL 解析失败不得报告为 API/HTTP 错误。

JSONL / 结果 URL 获取不得向预签名 URL、对象存储 URL 或其他 PaddleOCR 官方 API 源站之外的结果下载 URL 发送 Authorization 头。Authorization 仅用于 PaddleOCR 官方 API 请求。

已完成任务缺少结果 URL 是 `ResponseFormatError`。

未知任务状态是 `ResponseFormatError`，除非服务文档明确提供扩展机制，并且 SDK 文档说明未知状态如何表示。

## 发布验收清单

首个公开 SDK 版本发布前，实施团队必须完成以下检查：

- Python、TypeScript 和 Go 的公开客户端选项名称符合本契约。
- 缺少 `PADDLEOCR_ACCESS_TOKEN` 回退值时，认证客户端构造阶段失败，除非未来增加显式无认证模式。
- Base URL 规范化能一致处理尾部斜杠。
- 请求超时和轮询超时均已配置并测试为独立行为。
- 操作名称和阻塞行为符合操作表。
- `get_status`、`getStatus` 和 `GetStatus` 只执行非阻塞状态检查。
- 不存在名为 `get_result`、`getResult` 或 `GetResult` 的状态 API，且不保留旧草稿别名或单一 timeout 兼容选项。
- `Job`、`JobStatus`、`OCRResult` 和 `DocParsingResult` 在各语言中类型化并已文档化。
- `OCRResult` 和 `DocParsingResult` 暴露本契约要求的最小结构化字段，而不是只提供原始载荷。
- 任务状态在语言支持时类型化或枚举化，并具备 `pending`、`running`、`done`、`failed` 的基线语义。
- OCR API 暴露类型化模型参数，默认 PP-OCRv5，并通过集中的 OCR / 文档解析模型分类帮助函数校验。
- `done` 和 `failed` 是终态，未知状态产生 `ResponseFormatError`。
- 结果解析由显式任务/模型信息选择，而不是由 JSONL 字段是否存在推断。
- 等待方法以 `InvalidRequestError` 拒绝任务类型不匹配。
- 文件缺失优先级、SDK 语义校验、401/403、400、其他非 2xx、格式错误 2xx 和结果解析失败等错误优先级均已实现并测试。
- 格式错误的 2xx API 响应报告为 `ResponseFormatError`。
- 格式错误的 JSONL 载荷报告为 `ResultParseError`。
- 结果 URL 下载不会向预签名 / 对象存储 URL 发送 Authorization。
- 资源保存接受文档化的结果对象或 URL 输入，返回保存路径或类型化摘要，要求显式覆盖，并按文档映射下载 / 文件系统失败。
- 已完成任务缺少结果 URL 和未知任务状态均有测试覆盖。
- README 示例只使用契约批准的名称和行为。
- 包元数据、示例和生成文档与首个公开版本契约保持一致。
