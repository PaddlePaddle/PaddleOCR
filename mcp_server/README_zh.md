# PaddleOCR MCP 服务器

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mcp.svg)](https://pypi.org/project/mcp/)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)
[![OCR Engine](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)

轻量级 MCP (Model Context Protocol) 服务器，为所有 MCP Host 提供强大的 OCR 和文档分析功能。

**三种模式设计**：一套代码同时支持：
1. **星河/AI Studio 服务**：云端 API 调用
2. **本地服务化部署**：用户配置的 PaddleOCR 服务
3. **本地直接运行**：Python 环境直接调用 PaddleOCR

用户自主选择依赖管理策略，支持渐进式部署。

提供两个核心工具：
- `ocr_text(input_data)` - 文本识别，支持图像和 PDF
- `analyze_structure(input_data)` - 文档结构分析

输入支持：`http://`、`https://`、`file://`、`data:` URI，以及本地文件路径。

## 安装

### 基础安装（支持云端服务和本地服务化模式）

```bash
pip install mcp httpx numpy pillow
```

### 完整安装（支持本地直接运行模式）

```bash
# 基础依赖
pip install mcp httpx numpy pillow

# PaddleOCR 本地库（用户自行安装）
pip install paddlepaddle paddleocr opencv-python
```

## 使用

### STDIO 模式（默认）

```bash
python main.py
```

### HTTP 服务模式

```bash
python main.py --http --host 127.0.0.1 --port 3001
```

## MCP Host 配置

### 星河/AI Studio 模式（推荐 - 快速开始）

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["/absolute/path/to/main.py"],
      "cwd": "/absolute/path/to/mcp_server",
      "env": {
        "PADDLEOCR_MCP_OCR_SOURCE": "aistudio",
        "PADDLEOCR_MCP_API_URL": "https://your-api-endpoint.com/ocr",
        "PADDLEOCR_MCP_API_TOKEN": "your_token_here"
      }
    }
  }
}
```

### 本地服务化模式

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["/absolute/path/to/main.py"],
      "cwd": "/absolute/path/to/mcp_server",
      "env": {
        "PADDLEOCR_MCP_OCR_SOURCE": "user_service",
        "PADDLEOCR_MCP_API_URL": "http://your-service-host:8080/ocr"
      }
    }
  }
}
```

### 本地直接运行模式

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python", 
      "args": ["/absolute/path/to/main.py"],
      "cwd": "/absolute/path/to/mcp_server"
    }
  }
}
```

**重要**：将路径替换为实际的绝对路径。

## 环境变量

- `PADDLEOCR_MCP_OCR_SOURCE` - OCR 来源：`local`（默认）、`aistudio`、`user_service`
- `PADDLEOCR_MCP_API_URL` - API 服务地址
- `PADDLEOCR_MCP_API_TOKEN` - API 认证令牌
- `PADDLEOCR_MCP_TIMEOUT` - 超时时间（秒，默认 30）

## 调试

使用 MCP Inspector 调试：

```bash
npx @modelcontextprotocol/inspector
```

选择 `STDIO` 传输类型，命令输入 `python /path/to/main.py`，点击连接。

## 架构特点

- **渐进式依赖**：云端模式零配置，本地模式可选安装 PaddleOCR
- **统一接口**：三种模式使用相同的工具调用方式
- **智能切换**：运行时自动检测可用 OCR 来源
- **跨平台兼容**：支持 Windows、macOS、Linux

## 故障排除

检查 MCP Host 日志，如 Claude Desktop：
- **macOS**: `~/Library/Logs/Claude/mcp*.log`
- **Windows**: `%APPDATA%\Claude\logs\mcp*.log`

常见问题：
- 确认路径为绝对路径
- 云端模式检查网络连接和令牌
- 本地模式确认 PaddleOCR 安装

## 许可证

MIT License

---

**基于 [FastMCP v2](https://gofastmcp.com) 构建**
