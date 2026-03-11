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
- 如果运行在 AI 应用中，优先使用该应用自身的 secrets 或环境变量配置机制。

## 快速开始

1. 安装对应 skill 的依赖。
2. 推荐先在 shell、宿主应用或 secret manager 中设置所需环境变量。如果运行环境已经注入这些值，脚本会直接使用。
3. 本地调试和 smoke test 时，可以使用辅助脚本或共享的本地兜底文件：
   ```bash
   python skills/paddleocr-text-recognition/scripts/configure.py
   python skills/paddleocr-doc-parsing/scripts/configure.py
   cp skills/.env.example skills/.env
   ```
   然后按需填写 `skills/.env`。`skills/.env` 只是共享的本地兜底配置，不建议作为生产环境的默认配置方式。
4. 运行冒烟测试：

```bash
python skills/paddleocr-text-recognition/scripts/smoke_test.py
python skills/paddleocr-doc-parsing/scripts/smoke_test.py
```

## 文档入口

- 文本识别：`skills/paddleocr-text-recognition/SKILL.md`
- 文档解析：`skills/paddleocr-doc-parsing/SKILL.md`

## API 获取

请在 PaddleOCR 官网获取 API 信息：<https://www.paddleocr.com>

## 许可证

[Apache License 2.0](../LICENSE)
