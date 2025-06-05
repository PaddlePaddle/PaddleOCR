# PaddleOCR MCP Server

A Model Context Protocol (MCP) server for PaddleOCR document processing with three deployment modes.

## Installation & Setup

### 1. Install Dependencies
```bash
git clone <repository-url>
cd paddleocr-mcp-server
pip install -r requirements.txt

# For local mode only:
pip install paddleocr>=2.7.0
```

### 2. Configure Claude Desktop

Edit your config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Replace `/absolute/path/to/paddleocr-mcp-server` with your actual project path**

#### Local Mode (Offline)
```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["/absolute/path/to/paddleocr-mcp-server/main.py"],
      "cwd": "/absolute/path/to/paddleocr-mcp-server"
    }
  }
}
```

#### AI Studio Mode (Cloud API)
```json
{
  "mcpServers": {
    "paddleocr-aistudio": {
      "command": "python",
      "args": ["/absolute/path/to/paddleocr-mcp-server/main.py"],
      "cwd": "/absolute/path/to/paddleocr-mcp-server",
      "env": {
        "PADDLEOCR_ENGINE": "aistudio",
        "PADDLEOCR_API_URL": "https://xxx.aistudio-hub.baidu.com/ocr",
        "PADDLEOCR_API_TOKEN": "your_api_token_here"
      }
    }
  }
}
```

Get your token: [Baidu AI Studio](https://aistudio.baidu.com/index/accessToken)

#### Local Service Mode
```json
{
  "mcpServers": {
    "paddleocr-service": {
      "command": "python",
      "args": ["/absolute/path/to/paddleocr-mcp-server/main.py"],
      "cwd": "/absolute/path/to/paddleocr-mcp-server",
      "env": {
        "PADDLEOCR_ENGINE": "local_service",
        "PADDLEOCR_API_URL": "http://your-server:8080/ocr"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

## Available Tools

### `ocr_text(input_path, output_mode="simple")`
Extract text from images and PDFs.

### `analyze_structure(input_path, output_mode="detailed")`
Analyze document structure and layout.

## Input Support
- File paths: `/path/to/document.pdf`
- URLs: `https://example.com/image.jpg`
- Base64 data

## Output Modes
- **Simple (L1)**: Clean text/markdown for AI
- **Detailed (L2)**: JSON with coordinates and metadata

## Troubleshooting

Check Claude Desktop logs:
- **macOS**: `~/Library/Logs/Claude/mcp*.log`
- **Windows**: `%APPDATA%\Claude\logs\mcp*.log`

Verify absolute paths in your config file.

## License
MIT License 