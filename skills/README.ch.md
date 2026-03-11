# PaddleOCR Skills

本目录提供 PaddleOCR 官方 API 的 Agent Skills。

## 技能列表

- `paddleocr-text-recognition`：图片/PDF 文本识别。
- `paddleocr-doc-parsing`：版面感知文档解析。

## 支持模型

- `paddleocr-doc-parsing`：`PP-StructureV3`、`PaddleOCR-VL`、`PaddleOCR-VL-1.5`
- `paddleocr-text-recognition`：`PP-OCRv5`

## 所需环境变量

- `paddleocr-text-recognition`：`PADDLEOCR_OCR_API_URL`、`PADDLEOCR_ACCESS_TOKEN`
  可选：`PADDLEOCR_TIMEOUT`
- `paddleocr-doc-parsing`：`PADDLEOCR_DOC_PARSING_API_URL`、`PADDLEOCR_ACCESS_TOKEN`
  可选：`PADDLEOCR_DOC_PARSING_TIMEOUT`

## 前置条件

- 已安装 3.8 及以上版本的 Python，并可直接调用 `python`、`pip`。

## 快速开始

下面的示例覆盖两个 skill；只需执行你实际使用的那个 skill 对应命令。

1. 安装对应 skill 的依赖。
   ```bash
   python -m pip install -r paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements.txt
   # 第三行可选，仅在使用文档文件优化时需要
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```
2. 使用以下任一方式配置 API 凭证。

   方式 A：运行要测试的 skill 对应的辅助脚本。
   ```bash
   python paddleocr-text-recognition/scripts/configure.py
   python paddleocr-doc-parsing/scripts/configure.py
   ```

   方式 B：由 `.env.example` 模板文件创建本地 `.env` 文件，并填写所需变量。
   ```bash
   cp .env.example .env
   ```

   如果 skill 已安装到宿主应用目录（例如 `~/.claude/skills`），不要在那里运行 `configure.py` 或创建 `.env` 文件；应遵循宿主应用推荐的环境变量配置方式。
3. 运行要验证的 skill 对应的冒烟测试：

```bash
python paddleocr-text-recognition/scripts/smoke_test.py
python paddleocr-doc-parsing/scripts/smoke_test.py
```

## 在 AI 应用（如 Claude Code）中如何使用

可以直接用自然语言描述 OCR 或文档解析需求，并附上文件 URL 或本地路径，让 AI 应用调用对应 skill。

### paddleocr-text-recognition

URL 示例：
```text
提取这个文件中的全部文本：https://example.com/invoice.jpg
```

本地文件示例：
```text
提取本地文件 C:\docs\invoice.pdf 中的全部文本。
```

### paddleocr-doc-parsing

解析 URL 示例：
```text
解析这个 PDF，并返回主体内容和全部表格（结构化输出）：https://example.com/report.pdf
```

解析本地文件示例：
```text
解析本地文件 C:\docs\report.pdf，并返回完整结构化结果。
```

## 验证与排错

- 缺依赖报错：重新执行对应 requirements 文件的安装命令，例如 `paddleocr-text-recognition/scripts/requirements.txt`、`paddleocr-doc-parsing/scripts/requirements.txt`，以及文档文件优化需要时的 `paddleocr-doc-parsing/scripts/requirements-optimize.txt`。
- 配置问题：优先检查宿主应用或当前运行环境中是否已正确设置所需环境变量。
- 对于仓库内的冒烟测试，可以重跑对应的 `configure.py`，或更新本地 `.env` 文件。
- 如果使用 PaddleOCR 官方 API，可参考：<https://www.paddleocr.com>

## 文档入口

- 文本识别：`paddleocr-text-recognition/SKILL.md`
- 文档解析：`paddleocr-doc-parsing/SKILL.md`

## API 获取

如果使用 PaddleOCR 官方 API，可参考：<https://www.paddleocr.com>

## 许可证

[Apache License 2.0](../LICENSE)
