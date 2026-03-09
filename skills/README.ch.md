# PaddleOCR Skills

本目录提供 PaddleOCR 官方 API 的 Agent Skills。

## 技能列表

- `paddleocr-text-recognition`：图片/PDF 文本识别。
- `paddleocr-doc-parsing`：版面感知文档解析。

## 支持模型

- `paddleocr-doc-parsing`：`PP-StructureV3`、`PaddleOCR-VL`、`PaddleOCR-VL-1.5`
- `paddleocr-text-recognition`：`PP-OCRv5`

## 快速开始

此流程依赖 Node.js、`npm` 和 `npx`。如果本机没有 `npx`，请先安装 Node.js。

1. 先查看仓库内可安装的 skill：
   ```bash
   npx skills add PaddlePaddle/PaddleOCR --list
   ```
2. 全局安装目标 skill。下面示例同时安装两个 skill；如果只需要其中一个，也可以只安装对应 skill：
   ```bash
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
   ```
3. 验证安装结果：
   ```bash
   npx skills list -g
   ```
4. 安装 Python 依赖：
   ```bash
   python -m pip install -r ~/.agents/skills/paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements.txt
   # 第三行可选，仅在使用文档文件优化时需要
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```
   如果使用 Windows PowerShell，等价路径写法为：`$HOME\\.agents\\skills\\...`。
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

URL 示例：
```text
解析这个 PDF，并返回主体内容和全部表格（结构化输出）：https://example.com/report.pdf
```

本地文件示例：
```text
解析本地文件 C:\docs\report.pdf，并返回完整结构化结果。
```

## 验证与排错

- 检查是否安装成功：执行 `npx skills list -g`，确认需要的 skill 已安装。
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
