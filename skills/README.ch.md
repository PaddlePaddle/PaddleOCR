# PaddleOCR Skills

本目录提供 PaddleOCR 官方 API 的 Agent Skills。

## 技能列表

- `paddleocr-text-recognition`：图片/PDF 文本识别。
- `paddleocr-doc-parsing`：版面感知文档解析。

## 快速开始

1. 安装对应 skill 的依赖。
2. 通过脚本交互式配置 API 凭证：
   ```bash
   python skills/paddleocr-text-recognition/scripts/configure.py
   ```
   或手动复制 `.env.example` 为 `.env` 并填入凭证：`cp skills/.env.example skills/.env`
3. 运行冒烟测试：

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
