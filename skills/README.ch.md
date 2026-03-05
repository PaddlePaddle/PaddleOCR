# PaddleOCR Skills

本目录提供面向新手的 PaddleOCR 官方 API Agent Skills。只看本文档即可完成安装、配置与首跑。

## 技能列表

- `paddleocr-text-recognition`：图片/PDF 文本识别。
- `paddleocr-doc-parsing`：版面感知文档解析。

## 支持模型

- `paddleocr-doc-parsing`：`PP-StructureV3`、`PaddleOCR-VL`、`PaddleOCR-VL-1.5`
- `paddleocr-text-recognition`：`PP-OCRv5`
- 说明：实际模型能力与支持的文件格式取决于配置的 API 端点。

## 快速开始（npx）

1. 先查看仓库内可安装的 skill：
   ```bash
   npx skills add PaddlePaddle/PaddleOCR --list
   ```
2. 全局安装目标 skill：
   ```bash
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
   ```
3. 验证安装结果：
   ```bash
   npx skills list -g
   ```
4. 安装 Python 依赖（安装后立刻执行）：
   ```bash
   python -m pip install -r ~/.agents/skills/paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements.txt
   # Optional: required only when using document file optimization
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```
   Windows 等价路径写法：`$HOME\\.agents\\skills\\...`。
5. 通过脚本交互式配置 API 凭证：
   ```bash
   python ~/.agents/skills/paddleocr-text-recognition/scripts/configure.py
   python ~/.agents/skills/paddleocr-doc-parsing/scripts/configure.py
   ```
   共享配置文件位置：`~/.agents/skills/.env`
6. 运行冒烟测试（最小验收）：
   ```bash
   python ~/.agents/skills/paddleocr-text-recognition/scripts/smoke_test.py
   python ~/.agents/skills/paddleocr-doc-parsing/scripts/smoke_test.py
   ```

## 在聊天中如何使用

聊天请求可传 URL 或本地文件路径。

### 文本识别（`paddleocr-text-recognition`）

可直接复制发送：
```bash
Extract all text from this file: https://example.com/invoice.jpg
```

或：
```bash
Extract all text from local file C:\docs\invoice.pdf
```

### 文档解析（`paddleocr-doc-parsing`）

可直接复制发送：
```bash
Parse this PDF and return the main body plus all tables in structured format: https://example.com/report.pdf
```

或：
```bash
Parse local file C:\docs\report.pdf and return complete structured output.
```

## 验证与排错

- 检查是否安装成功：执行 `npx skills list -g`。
- 缺依赖报错：重新执行对应的 `python -m pip install -r ...`。
- 配置报错：重跑对应 skill 的 `configure.py`。
- API 地址与 token 获取入口：<https://www.paddleocr.com>

## 文档入口

- 文本识别：`skills/paddleocr-text-recognition/SKILL.md`
- 文档解析：`skills/paddleocr-doc-parsing/SKILL.md`

## API 获取

请在 PaddleOCR 官网获取 API 信息：<https://www.paddleocr.com>

## 许可证

[Apache License 2.0](../LICENSE)
