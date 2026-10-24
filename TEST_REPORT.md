# PaddleOCR API SDK 集成测试报告

**测试分支**: feature/api-sdk (PR #18049)  
**测试环境**: macOS Darwin 24.3.0, Python 3.9.6, requests 2.32.5  
**API Endpoint**: https://paddleocr.aistudio-app.com/api/v2/ocr/jobs

---

## 测试结果总览

| #   | 测试项　　　　　　　　　　　　　　　　| 结果　 | 耗时 | 说明　　　　　　　　　　　　　　　　　　　　　　　|
| -----| ---------------------------------------| --------| ------| ---------------------------------------------------|
| 1   | OCR URL (PP-OCRv5)　　　　　　　　　　| ✅ PASS | 3.6s | URL 输入，默认参数　　　　　　　　　　　　　　　　|
| 2   | OCR URL + 自定义 Options　　　　　　　| ✅ PASS | 3.6s | 设置 use_doc_orientation_classify=True　　　　　　|
| 3   | Doc Parsing URL (PP-StructureV3)　　　| ✅ PASS | 3.7s | 文档版面解析　　　　　　　　　　　　　　　　　　　|
| 4   | Submit + Poll 分步调用　　　　　　　　| ✅ PASS | 3.7s | 非阻塞 API：submit → get_result → wait_for_result |
| 5   | OCR 本地文件上传　　　　　　　　　　　| ✅ PASS | 4.3s | file_path 模式　　　　　　　　　　　　　　　　　　|
| 6   | 错误处理 (无效 token)　　　　　　　　 | ✅ PASS | 0.2s | 正确抛出 AuthError　　　　　　　　　　　　　　　　|
| 7   | 输入校验　　　　　　　　　　　　　　　| ✅ PASS | 0.0s | 缺少输入 / 互斥参数均正确拦截　　　　　　　　　　 |
| 8   | Context Manager (with)　　　　　　　　| ✅ PASS | 3.6s | with 语句正常工作　　　　　　　　　　　　　　　　 |
| 9   | PaddleOCR-VL 模型　　　　　　　　　　 | ✅ PASS | 3.6s | VL 模型正常返回 markdown　　　　　　　　　　　　　|
| 10  | PaddleOCR-VL-1.5 模型　　　　　　　　 | ✅ PASS | 3.6s | VL-1.5 模型正常返回 markdown　　　　　　　　　　　|
| 11  | Doc Parsing 文件上传 (PP-StructureV3) | ✅ PASS | 4.3s | 本地文件上传 + 文档解析　　　　　　　　　　　　　 |

**总计: 11 passed, 0 failed**

---

## 发现的 Bug（已修复验证）

### 🔴 阻塞性 Bug: `fetch_jsonl` 请求 BOS 签名 URL 时携带了多余的 Authorization header

**文件**: `paddleocr/_api_client/_http.py` (同步) + `async_client.py` (异步)

**现象**: 任务提交和轮询均成功，但在最后一步下载 JSONL 结果文件时，SDK 使用带有 `Authorization: bearer <paddle_token>` 的 session 去请求百度 BOS 对象存储的预签名 URL。BOS 不认识这个 header，返回 400 Bad Request。

**根因**: `HTTPClient.fetch_jsonl()` 使用 `self._session.get(url)` 发起请求，而 session 在初始化时设置了 `Authorization` header。BOS 的预签名 URL 已经包含了自己的鉴权参数（`authorization=bce-auth-v1/...`），额外的 Authorization header 导致冲突。

**修复**: `fetch_jsonl` 应使用不带 auth header 的独立请求：
```python
# 修复前
resp = self._session.get(url, timeout=self._timeout)

# 修复后
resp = requests.get(url, timeout=self._timeout)
```

**影响范围**: 所有实际 API 调用（OCR、Doc Parsing、所有模型）在修复前均无法获取结果。这是一个 **必须在合入前修复的阻塞性 bug**。

---

## 其他已知问题（非阻塞，可后续迭代）

| 严重度 | 语言 | 问题 |
|--------|------|------|
| 中 | Python | AsyncAPIClient._poll_until_done 使用硬编码 DEFAULT_MAX_WAIT_TIME，忽略用户设置的 timeout |
| 中 | Go | submitURL/submitFile/getJobStatus 未使用 http.NewRequestWithContext，context 取消无法中断 HTTP 请求 |
| 中 | TypeScript | poller.ts 的 sleep 方法 abort listener 未设置 `{ once: true }`，长轮询会泄漏 listener |
| 中 | TypeScript | http.ts 的 fetchJsonl 未检查 resp.ok（同样的 BOS auth 问题可能也存在于 TS SDK） |
| 低 | 全部 | 轮询循环先 sleep 再 check，对已完成任务多等 3 秒 |
| 低 | Python | CLI argparse 的 store_true 传 False 而非 None，导致多余字段发送给 API |

---

## 结论

**修复 `fetch_jsonl` 的 auth header 问题后，Python SDK 的核心功能全部正常工作。** 4 个模型（PP-OCRv5、PP-StructureV3、PaddleOCR-VL、PaddleOCR-VL-1.5）均可正常调用，URL 输入和文件上传两种模式均可用，错误处理和输入校验逻辑正确。

**建议**: 
1. ⚠️ **合入前必须修复** `fetch_jsonl` 的 BOS auth header bug（Python 同步 + 异步，以及 Go/TypeScript SDK 中的同类问题）
2. 其他问题可以合入后迭代修复
