# PaddleOCR Skills

本目录提供 PaddleOCR 官方 API 的 Agent Skills。

## 技能列表

- `paddleocr-text-recognition`：图片/PDF 文本识别。
- `paddleocr-doc-parsing`：版面感知文档解析。

## 所需环境变量

- `paddleocr-text-recognition`：`PADDLEOCR_OCR_API_URL`、`PADDLEOCR_ACCESS_TOKEN`
  可选：`PADDLEOCR_TIMEOUT`
- `paddleocr-doc-parsing`：`PADDLEOCR_DOC_PARSING_API_URL`、`PADDLEOCR_ACCESS_TOKEN`
  可选：`PADDLEOCR_DOC_PARSING_TIMEOUT`

## 前置条件

- 需要本机已安装并可直接调用 `python`、`pip`。
- 本地辅助命令示例默认 shell 环境可执行 `cp` 等基础命令。
- 如果 skill 安装在宿主应用目录中，请遵循该宿主应用的环境变量配置最佳实践，不要在那里创建本地配置文件。

## 快速开始

以下命令默认在 `skills/` 目录下执行。

1. 安装对应 skill 的依赖。
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

## 文档入口

- 文本识别：`paddleocr-text-recognition/SKILL.md`
- 文档解析：`paddleocr-doc-parsing/SKILL.md`

## API 获取

请在 PaddleOCR 官网获取 API 信息：<https://www.paddleocr.com>

## 许可证

[Apache License 2.0](../LICENSE)
