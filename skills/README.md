# PaddleOCR Claude Code Skills

中文 | [English](./README_en.md)

## 简介

本目录提供两个 Claude Code 技能，通过 PaddleOCR 官方 API 实现 OCR 文字识别与文档解析功能。

| 技能 | 用途 | 适用场景 |
|------|------|---------|
| **paddleocr-text-recognition** | 快速文字提取 | 截图、扫描件、简单文档 |
| **paddleocr-doc-parsing** | 高级文档解析 | 表格、公式、印章、复杂版面文档 |

## 与 MCP Server 的关系

本目录的 Skills 与 `mcp_server/` 互补，两者调用相同的底层模型，但服务于不同场景：

| 特性 | MCP Server | Skills |
|------|-----------|--------|
| 协议 | Model Context Protocol (MCP) | Claude Code Skill Protocol |
| 客户端 | Claude Desktop、VSCode 等 | Claude Code CLI |
| 架构 | 长驻服务进程 | 直接 CLI 调用 |

## 安装

### 1. 安装依赖

```bash
pip install -r skills/paddleocr-text-recognition/scripts/requirements.txt
pip install -r skills/paddleocr-doc-parsing/scripts/requirements.txt
```

### 2. 获取 API 凭证

访问 [https://paddleocr.com](https://paddleocr.com) 获取 API URL 和 Access Token。

### 3. 配置

```bash
python skills/paddleocr-text-recognition/scripts/configure.py
python skills/paddleocr-doc-parsing/scripts/configure.py
```

### 4. 验证

```bash
python skills/paddleocr-text-recognition/scripts/smoke_test.py
python skills/paddleocr-doc-parsing/scripts/smoke_test.py
```

## 使用

### paddleocr-text-recognition 文字识别

```bash
# URL 输入
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-url "https://example.com/image.png"

# 本地文件
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-path "./document.pdf" --pretty
```

### paddleocr-doc-parsing 文档解析

```bash
# URL 输入
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-url "https://example.com/doc.pdf"

# 本地文件
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-path "./invoice.pdf" --pretty
```

## 目录结构

```
skills/
├── README.md                          # 中文说明
├── README_en.md                       # 英文说明
├── paddleocr-text-recognition/        # 文字识别
│   ├── SKILL.md                       # 技能定义
│   ├── scripts/                       # Python 脚本
│   └── references/                    # API 参考文档
└── paddleocr-doc-parsing/             # 文档解析
    ├── SKILL.md                       # 技能定义
    ├── scripts/                       # Python 脚本
    └── references/                    # API 参考文档
```
