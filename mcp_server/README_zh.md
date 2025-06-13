# PaddleOCR MCP 服务器

[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)

本项目提供一个轻量级的 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/introduction) 服务器，旨在将 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的强大能力集成到任何兼容的 MCP Host 中。

### 主要功能
- **当前支持的工具**
    - **通用文本识别**: 支持对图像和PDF文件进行高精度的文本检测与识别 (`PP-OCRv5`)。
    - **版面分析**: 支持从图像和PDF中提取文本块、标题、段落、图片、表格等结构化信息 (`PP-StructureV3`)。
- **支持运行在如下工作模式**:
    - **本地 Python 库**: 在本机直接运行 PaddleOCR 模型。
    - **AI Studio 星河社区服务**: 调用飞桨 AI Studio 提供的云端服务。
    - **自托管服务**: 调用用户自行部署的 PaddleOCR 服务。

### 目录

- [1. 安装](#1-安装)
- [2. 快速开始](#2-快速开始)
- [3. 配置说明](#3-配置说明)
  - [3.1. MCP Host 配置](#31-mcp-host-配置)
  - [3.2. 工作模式详解](#32-工作模式详解)
    - [模式一：托管在AI Studio 星河社区的服务](#模式一托管在ai-studio-星河社区的服务-aistudio)
    - [模式二：本地 Python 库](#模式二本地-python-库-local)
    - [模式三：自托管服务](#模式三自托管服务-self_hosted)
- [4. 参数参考](#4-参数参考)
- [5. 配置示例](#5-配置示例)
  - [5.1. AI Studio 星河社区服务配置](#51-ai-studio-星河社区服务配置)
  - [5.2. 本地 Python 库配置](#52-本地-python-库配置)
  - [5.3. 自托管服务配置](#53-自托管服务配置)

## 1. 安装

```bash
# 从 PyPI 安装
pip install paddleocr_mcp

# 或者，从项目源码安装
pip install -e .
```

根据您选择的 [工作模式](#32-工作模式详解)，可能需要安装额外依赖。

## 2. 快速开始

本节将以 **Claude Desktop** 作为 MCP Host，并使用 **AI Studio 星河社区服务**为例，引导您完成快速配置。此模式无需在本地安装复杂的依赖，推荐新用户使用。

1.  **准备 AI Studio 星河社区服务**
    -   访问 [飞桨AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine) 并登录。
    -   在左侧更多内容下的 **PaddleX产线** 部分，- [创建产线] - [ OCR ] - [通用 OCR ] - [直接部署] - [文本识别模块 选择PP-OCRv5_server_rec] - [开始部署]
    -   部署成功后，获取您的 *API地址* (示例："https://xxxxxx.aistudio-hub.baidu.com")。
    -   获取您的 *访问令牌*，[通过 https://aistudio.baidu.com/index/accessToken 获取](https://aistudio.baidu.com/index/accessToken)

2.  **定位 MCP 配置文件** - 详情请参考[官方开始文档](https://modelcontextprotocol.io/quickstart/user)
    -   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    -   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    -   **Linux**: `~/.config/Claude/claude_desktop_config.json`

3.  **添加MCP服务器配置**
    打开 `claude_desktop_config.json` 文件，参考 [5.1. AI Studio 星河社区服务配置](#51-ai-studio-星河社区服务配置), 复制其内容到 `claude_desktop_config.json` 中。

    **注意**: 请务必填入您自己的*访问令牌*与*API地址*。如果 `paddleocr_mcp` 命令不在系统 PATH 中，请将 `command` 设置为可执行文件的绝对路径。

4.  **重启 MCP Host**
    重启 Claude Desktop。新的 `PaddleOCR` 工具现在应该可以在应用中使用了。


## 3. 配置说明

### 3.1. MCP Host 配置

在 Host 的配置文件中（如 `claude_desktop_config.json`），您需要定义工具服务器的启动方式。关键字段如下：
-   `command`: `paddleocr_mcp` 可执行文件名（如果在 PATH 中）或绝对路径。
-   `args`: 命令行参数数组，可以为空或添加需要的参数（如 `["--verbose"]`）。
-   `env`: 环境变量配置。

### 3.2. 工作模式详解

您可以根据需求，使 MCP 服务器运行在不同的工作模式。除了使用系统内置的模型方案外，您还可以替换为自己训练的自定义模型。

#### 模式一：托管在AI Studio 星河社区的服务 (`aistudio`)
此模式调用 [飞桨 AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine) 的服务。
-   **适用场景**: 免本地安装，快速体验功能，测试等。
-   **准备**:
    1.  按照 [快速开始](#2-快速开始) 的指引，部署服务并获取 *API地址* 和 *访问令牌*。
    2.  在配置中设置环境变量 `PADDLEOCR_MCP_PPOCR_SOURCE` 为 `"aistudio"`。
    3.  将获取到的 *API地址* 和 *访问令牌* 填入 `PADDLEOCR_MCP_SERVER_URL` 和 `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`。

#### 模式二：本地 Python 库 (`local`)
此模式直接在本地计算机上运行模型，对本地环境与计算机性能有一定要求。
-   **适用场景**: 需要离线使用、对数据隐私有严格要求、需要大量批处理任务的场景。
-   **准备**:
    1.  必须安装 *飞桨框架* 和 *PaddleOCR*。为避免依赖冲突，**强烈建议在独立的虚拟环境中安装**。
    2.  请遵循 **[PaddleOCR 官方安装文档](./PaddleOCR/docs/version3.x/installation.md)** 进行安装。
    3.  参考 [配置示例](#52-本地-python-库配置) 配置 `claude_desktop_config.json` 文件。
    4.  如您需要进行OCR参数调整或二次开发，请参考[PaddleOCR 官方文档](https://paddlepaddle.github.io/PaddleOCR/latest/) 

#### 模式三：自托管服务 (`self_hosted`)
此模式调用您自行部署的 PaddleOCR 推理服务。
-   **适用场景**: 生产环境部署、需要高并发处理、需要自定义模型配置的企业级应用场景，具备服务化部署优势及高度灵活性。
-   **准备**:
    1.  同上，安装 *飞桨框架* 和 *PaddleOCR*。
    2.  请参考 **[ PaddleOCR 服务化部署文档](https://paddlepaddle.github.io/PaddleOCR/main/version3.x/deployment/serving.html)** 完成服务的部署。
    3.  参考 [配置示例](#53-自托管服务配置) 配置 `claude_desktop_config.json` 文件。
    4.  将您的服务地址填入 `PADDLEOCR_MCP_SERVER_URL` (例如: `"http://127.0.0.1:8000"`)。


## 4. 参数参考
您可以通过环境变量或命令行参数来控制服务器的行为。

| 环境变量                               | 命令行参数                    | 参数类型          | 描述                                                       | 可选值                                     | 默认值            |
| -------------------------------------- | ----------------------------- | ------------- | ---------------------------------------------------------- | ------------------------------------------ | ----------------- |
| `PADDLEOCR_MCP_PIPELINE`               | `--pipeline`                  | `str`         | 选择要运行的产线。                                         | `"OCR"`, `"PP-StructureV3"`                | `"OCR"`           |
| `PADDLEOCR_MCP_PPOCR_SOURCE`           | `--ppocr_source`              | `str`         | 选择 PaddleOCR 的能力来源。                                | `"local"`, `"aistudio"`, `"self_hosted"`   | `"local"`         |
| `PADDLEOCR_MCP_SERVER_URL`             | `--server_url`                | `str`         | 底层服务 URL (`aistudio` 或 `self_hosted` 模式下必需)。    | -                                          | `None`            |
| `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`  | `--aistudio_access_token`     | `str`         | AI Studio 认证令牌 (`aistudio` 模式下必需)。               | -                                          | `None`            |
| `PADDLEOCR_MCP_TIMEOUT`                | `--timeout`                   | `int`         | 底层服务请求的超时时间（秒）。                                 | -                                          | `30`              |
| `PADDLEOCR_MCP_DEVICE`                 | `--device`                    | `str`         | 指定运行推理的设备 (仅在 `local` 模式下生效)。             | `"cpu"`, `"gpu:0"`, ...                    | `"cpu"`           |
| `PADDLEOCR_MCP_PIPELINE_CONFIG`        | `--pipeline_config`           | `str`         | PaddleOCR 产线配置文件路径 (仅在 `local` 模式下生效)。      | -                                          | `None`            |
| -                                      | `--http`                      | `bool`        | 使用 HTTP 传输而非 STDIO (适用于远程部署和多客户端)。       | -                                          | `False`           |
| -                                      | `--host`                      | `str`         | HTTP 模式的主机地址。                                       | -                                          | `"127.0.0.1"`     |
| -                                      | `--port`                      | `int`         | HTTP 模式的端口。                                           | -                                          | `8000`            |
| -                                      | `--verbose`                   | `bool`        | 启用详细日志记录用于调试。                                  | -                                          | `False`           |

## 5. 配置示例

以下是针对不同工作模式的完整配置示例，您可以直接复制并根据需要修改：

### 5.1. AI Studio 星河社区服务配置

```json
{
  "mcpServers": {
    "paddleocr-aistudio-ocr": {
      "command": "paddleocr_mcp",
      "args": [],
      "env": {
        "PADDLEOCR_MCP_PIPELINE": "OCR",
        "PADDLEOCR_MCP_PPOCR_SOURCE": "aistudio",
        "PADDLEOCR_MCP_SERVER_URL": "https://<your-aistudio-endpoint>", 
        "PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN": "your_aistudio_token_here"
      }
    }
  }
}
```

**使用说明**：
- 将 `<your-aistudio-endpoint>` 替换为您的 AI Studio 星河社区服务的 *API地址*，注意使用base url, 例如 **https://xxxxx.aistudio-hub.baidu.com/**
- 将 `your_aistudio_token_here` 替换为您的 *访问令牌*

### 5.2. 本地 Python 库配置

```json
{
  "mcpServers": {
    "paddleocr-local-ocr": {
      "command": "paddleocr_mcp",
      "args": [],
      "env": {
        "PADDLEOCR_MCP_PIPELINE": "OCR",
        "PADDLEOCR_MCP_PPOCR_SOURCE": "local"
      }
    }
  }
}
```

**使用说明**：
- `PADDLEOCR_MCP_PIPELINE_CONFIG` 为可选项，不设置时使用默认配置
- 如需自定义，请将路径替换为您的配置文件绝对路径。如何导出配置文件请参考[PaddleOCR 与 PaddleX 配置说明 3.1章节](https://paddlepaddle.github.io/PaddleOCR/main/version3.x/paddleocr_and_paddlex.html?h=%E5%AF%BC%E5%87%BA#31)。

### 5.3. 自托管服务配置

```json
{
  "mcpServers": {
    "paddleocr-self-hosted-ocr": {
      "command": "paddleocr_mcp",
      "args": [],
      "env": {
        "PADDLEOCR_MCP_PIPELINE": "OCR",
        "PADDLEOCR_MCP_PPOCR_SOURCE": "self_hosted",
        "PADDLEOCR_MCP_SERVER_URL": "http://<your-server-url>"
      }
    }
  }
}
```

**使用说明**：
- 将 `<your-service-host>` 替换为您底层服务的base URL（如：`http://127.0.0.1:8000`）
