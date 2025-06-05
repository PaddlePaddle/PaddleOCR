# PaddleOCR MCP Server v2.0

强大的 MCP (Model Context Protocol) 服务器，支持三种 OCR 处理模式：
- **本地模式**: 使用本地 PaddleOCR 库（离线、隐私保护）
- **星河API模式**: 调用百度星河云端 API（高性能）
- **本地服务模式**: 调用本地部署的 PaddleOCR 服务

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd mcp_server

# 安装基础依赖
pip install -r requirements.txt

# 本地模式需要额外安装
paddle paddle > 3.0.0
paddleocr>=3.0.0
```

### 2. 配置 Claude Desktop 示例

编辑配置文件:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

**将 `/absolute/path/to/PaddleOCR/mcp_server` 替换为实际项目路径**

#### 本地模式（离线处理）

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["/absolute/path/to/PaddleOCR/mcp_server/main.py"],
      "cwd": "/absolute/path/to/PaddleOCR/mcp_server"
    }
  }
}
```

#### 星河API模式（云端处理）

```json
{
  "mcpServers": {
    "paddleocr-aistudio": {
      "command": "python",
      "args": ["/absolute/path/to/PaddleOCR/mcp_server/main.py"],
      "cwd": "/absolute/path/to/PaddleOCR/mcp_server",
      "env": {
        "PADDLEOCR_ENGINE": "aistudio",
        "PADDLEOCR_API_URL": "https://xxx.aistudio-hub.baidu.com/ocr",
        "PADDLEOCR_API_TOKEN": "your_api_token_here"
      }
    }
  }
}
```

获取API Token: [百度AI Studio](https://aistudio.baidu.com/index/accessToken)

#### 本地服务模式

```json
{
  "mcpServers": {
    "paddleocr-service": {
      "command": "python",
      "args": ["/absolute/path/to/PaddleOCR/mcp_server/main.py"],
      "cwd": "/absolute/path/to/PaddleOCR/mcp_server",
      "env": {
        "PADDLEOCR_ENGINE": "local_service",
        "PADDLEOCR_API_URL": "your_url_here"
      }
    }
  }
}
```

### 3. 重启 Claude Desktop

## 🛠️ 可用工具

### `ocr_text(input_path, output_mode="simple")`
从图片和PDF中提取文字

### `analyze_structure(input_path, output_mode="detailed")`
分析文档结构（表格、公式、布局等）

## 📋 输入支持
- 文件路径: `/path/to/document.pdf`
- URL地址: `https://example.com/image.jpg`  
- Base64数据

## 📊 输出模式
- **Simple (L1)**: 简洁文本/Markdown，适合AI处理
- **Detailed (L2)**: JSON格式，包含坐标和元数据

## 🔍 故障排除

查看 Claude Desktop 日志:
- **macOS**: `~/Library/Logs/Claude/mcp*.log`
- **Windows**: `%APPDATA%\Claude\logs\mcp*.log`

确保配置文件中使用绝对路径。

## 📦 模式对比

| 功能特性 | 本地模式 | 星河API | 本地服务 |
|---------|---------|--------|----------|
| **隐私安全** | ✅ 完全离线 | ⚠️ 云端处理 | ✅ 自主部署 |
| **部署简便** | ❌ 需要模型文件 | ✅ 即开即用 | ⚠️ 需要服务 |
| **处理速度** | ⚠️ 取决于硬件 | ✅ 云端算力 | ✅ 可扩展 |
| **成本费用** | ✅ 免费使用 | ⚠️ API费用 | ✅ 仅基础设施 |

## 🔧 命令行用法

```bash
# 本地模式
python main.py --engine local

# 星河API模式  
python main.py --engine aistudio \
  --api-url https://xxx.aistudio-hub.baidu.com/ocr \
  --api-token your_token

# 本地服务模式
python main.py --engine local_service \
  --api-url http://your-server:8080/ocr
```

## 📄 许可证

MIT License

---

**基于 [FastMCP v2](https://gofastmcp.com) 构建** - 简洁优雅的 MCP 服务器开发框架
