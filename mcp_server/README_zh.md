# PaddleOCR MCP 服务器

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddleocr-mcp.svg)](https://pypi.org/project/paddleocr-mcp/)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)
[![OCR Engine](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)

本产品是一个轻量级的 [MCP (Model Context Protocol)](https://gofastmcp.com) 服务器，旨在将 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的强大能力集成到任何兼容的 MCP Host 应用中。

### 主要功能
- **通用文本识别**: 支持对图像和PDF文件进行高精度的文本检测与识别 (`PP-OCRv5`)。
- **版面分析**: 支持从图像和PDF中提取文本块、标题、段落、图片、表格等结构化信息 (`PP-StructureV3`)。
- **根据 PaddleOCR 能力来源的不同，支持三种工作模式**:
    - **本地服务**: 在本机直接运行 PaddleOCR 模型。
    - **AI Studio 服务**: 调用飞桨 AI Studio 提供的云端服务。
    - **服务化部署**: 调用用户自行部署的 PaddleOCR 服务。

## 1. 安装

```bash
# 从 PyPI 安装
pip install paddleocr-mcp

# 或者，从项目源码安装
pip install -e .
```

根据您选择的 [服务模式](#32-服务模式详解)，可能需要安装额外依赖。

## 2. 快速开始

本节将以 **Claude Desktop** 作为 MCP Host，并使用 **AI Studio 服务模式**为例，引导您完成快速配置。此模式无需在本地安装复杂的依赖，推荐新用户使用。

1.  **准备 AI Studio 服务**
    -   访问 [飞桨AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine) 并登录。
    -   在左侧更多内容下的 **PaddleX产线** 部分，- [创建产线] - [ OCR ] - [通用 OCR ] - [直接部署] - [文本识别模块 选择PP-OCRv5_server_rec] - [开始部署]
    -   部署成功后，获取您的 **API地址** (示例："https://xxxxxx.aistudio-hub.baidu.com/ocr") 和 **访问令牌 (Access Token)**。


2.  **定位 MCP 配置文件**
    -   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    -   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    -   **Linux**: `~/.config/Claude/claude_desktop_config.json`

3.  **添加工具配置**
    打开 `claude_desktop_config.json` 文件，从 `sample_configs` 目录中选择与您需求匹配的模板（例如 `aistudio_ocr.json`），并将其内容复制到 `claude_desktop_config.json` 中。

    **注意**: 请务必将 `command` 和 `cwd` 中的路径修改为您的本地**绝对路径**，并填入您自己的 AI Studio 服务信息。

    ```json
    {
        "mcpServers": {
            "paddleocr-aistudio-ocr": {
            "command": "/path/to/your/python_env/bin/python",
            "args": [
                "-m",
                "paddleocr_mcp"
            ],
            "cwd": "/path/to/your/PaddleOCR/mcp_server",
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

4.  **重启 MCP Host**
    重启 Claude Desktop。新的 `PaddleOCR` 工具现在应该可以在应用中使用了。


## 3. 配置指南

### 3.1. MCP Host 配置

在 Host 的配置文件中（如 `claude_desktop_config.json`），您需要定义工具服务器的启动方式。关键字段如下：
-   `command`: 指向您环境中 Python 解释器的**绝对路径**。
-   `cwd`: 指向 `mcp_server` 目录的**绝对路径**。

### 3.2. 服务模式详解

您可以根据需求选择不同的 PaddleOCR 服务。

#### 模式一：AI Studio 服务模式 (`aistudio`) *(推荐)*
此模式通过 API 调用 [飞桨 AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine) 的服务。
-   **优点**: 快速接入，无需本地复杂环境配置。
-   **准备**:
    1.  按照 [快速开始](#2-快速开始) 的指引，部署服务并获取 `API地址` 和 `访问令牌`。
    2.  在配置中设置 `PADDLEOCR_MCP_PPOCR_SOURCE="aistudio"`。
    3.  填入 `PADDLEOCR_MCP_SERVER_URL` 和 `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`。

#### 模式二：本地服务模式 (`local`)
此模式直接在本地计算机上运行模型，对本地环境有一定要求。
-   **优点**: 数据隐私性好，无网络依赖。
-   **准备**:
    1.  必须安装 `飞桨框架` 和 `PaddleOCR`。为避免依赖冲突，强烈建议使用独立的虚拟环境。
    2.  请严格遵循 **[ PaddleOCR 官方快速开始文档](https://paddlepaddle.github.io/PaddleOCR/main/quick_start.html)** 进行安装。
    3.  sample_configs 文件夹中配置示例，并在配置中设置相应的参数部分.

#### 模式三：服务化部署模式 (`self_hosted`)
此模式调用您自行部署的 PaddleOCR 推理服务。
-   **优点**: 兼具性能与灵活性，适合生产环境。
-   **准备**:
    1.  请参考 **[ PaddleOCR 服务化部署文档](https://paddlepaddle.github.io/PaddleOCR/main/version3.x/deployment/serving.html)** 完成服务的部署。
    2.  sample_configs 文件夹中配置示例，并在配置中设置需要的参数部分。
    3.  将您的服务地址填入 `PADDLEOCR_MCP_SERVER_URL` (例如: `"http://127.0.0.1:8000"`)。


## 4. 参数参考
您可以通过环境变量或命令行参数来控制服务器的行为。

| 环境变量                               | 描述                                                       | 可选值                                   | 默认值          |
| -------------------------------------- | ---------------------------------------------------------- | ---------------------------------------- | --------------- |
| `PADDLEOCR_MCP_PIPELINE`               | 选择要运行的产线。                                         | `"OCR"`, `"PP-StructureV3"`              | `"OCR"`         |
| `PADDLEOCR_MCP_PPOCR_SOURCE`           | 选择 PaddleOCR 的服务来源。                                | `"local"`, `"aistudio"`, `"self_hosted"` | `"local"`       |
| `PADDLEOCR_MCP_SERVER_URL`             | 后端服务 URL (`aistudio` 或 `self_hosted` 模式下必需)。    | -                                        | -               |
| `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`  | AI Studio 认证令牌 (`aistudio` 模式下必需)。               | -                                        | -               |
| `PADDLEOCR_MCP_TIMEOUT`                | API 请求的超时时间（秒）。                                 | -                                        | `30`            |
| `PADDLEOCR_MCP_DEVICE`                 | 指定运行推理的设备 (仅在 `local` 模式下生效)。             | `cpu`, `gpu:0`, ...                      | `cpu`           |
