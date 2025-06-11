# PaddleOCR MCP 服务器

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddleocr-mcp.svg)](https://pypi.org/project/paddleocr-mcp/)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)
[![OCR Engine](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)

一个轻量级的 [MCP (Model Context Protocol)](https://gofastmcp.com) 服务器，为所有 MCP Host提供强大的 OCR 和文档分析功能。

**提供工具**：
- `PP-OCRv5`：通用文本识别，支持图像和 PDF。
- `PP-Structurev3`：文档版面分析，支持解析表格、图片，公式识别等，支持图像和 PDF。
- 支持三种部署方式：
  - 本地部署：使用本地 PaddleOCR 库
  - AI Studio 部署：使用飞桨 AI Studio 星河社区服务
  - 服务化部署：使用自建的 PaddleOCR 服务器

## 1. 【安装核心包】

```bash
pip install paddleocr-mcp

# 或者，从项目源码安装
pip install -e .
```


## 2. 【配置MCP使用】

本服务器设计为由 MCP Host（如 Claude Desktop）启动和管理。你只需在 Host 的配置文件中添加一个条目即可。

### 配置步骤 (以 Claude Desktop 为例)

1.  **打开 Claude Desktop 的 MCP 配置文件**。 请参考 [MCP 快速开始文档](https://mcp-docs.cn/quickstart/user) 获取更多信息。
    -   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    -   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    -   **Linux**: `~/.config/Claude/claude_desktop_config.json`

2.  **选择一个配置模式**。 (详见 3.【部署PaddleOCR】) 
    从 `config_example` 目录中选择一个符合你需求的 `.json` 文件。该目录提供了所有支持的模式和产线的组合。

3.  **将示例配置复制到 `claude_desktop_config.json`**。
    例如，如果你想使用 **AI Studio** 的 **文档分析 (Structure)** 功能，可以复制 `config_example/aistudio_structure.json` 的内容

4.  **修改配置中的占位符**。
    -  `command`: 你的 Python 解释器**绝对路径**。
    -  `cwd`: `mcp_server` 目录的**绝对路径**。
    -   根据所选模式，填写 `PADDLEOCR_MCP_SERVER_URL` 和 `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN` 等。  

5.  **重启 Claude Desktop**。
    重启后，新的 `paddleocr` 工具即可在 Claude 中使用。

## 3. 【部署PaddleOCR】

#### AIStudio 星河社区部署 *（快速测试推荐）* 

1.  [飞桨AI Studio星河社区-人工智能学习与实训社区](https://aistudio.baidu.com/pipeline/mine)
2.  [点击左侧更多内容下的PaddleX产线](https://aistudio.baidu.com/pipeline/mine) - [创建产线] - [OCR] - [通用OCR] - [直接部署] - [文本识别模块 选择PP-OCRv5_server_rec] - [开始部署]
3.  找到自己部署的API 地址 （示例："https://xxxxxx.aistudio-hub.baidu.com/ocr"）添加到 claude_desktop_config.json 

- AI Studio Token密钥 (从 https://aistudio.baidu.com/index/accessToken 获取)
- (这里以”PP-OCRv5“和“直接部署”为例子，星河社区支持多种产线与模型训练)

#### 本地部署

如果你需要使用 `local` 模式，则必须安装 `paddlepaddle` 和 `paddleocr`。

具体的安装方法，请参考 **[PaddleOCR官方快速开始文档](https://paddlepaddle.github.io/PaddleOCR/main/quick_start.html)**，以确保版本和环境兼容。

#### 服务化部署 

如果你需要使用 `self_hosted` 模式，请参考 **[PaddleOCR 服务化部署文档](https://paddlepaddle.github.io/PaddleOCR/main/version3.x/deployment/serving.html)**。

找到自己部署的API 地址 （示例："http://127.0.0.1:8000"）添加到 claude_desktop_config.json 

服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleOCR 推荐用户使用 [PaddleX](https://github.com/PaddlePaddle/PaddleX) 进行服务化部署。请阅读 [PaddleOCR 与 PaddleX 的区别与联系](https://paddlepaddle.github.io/PaddleOCR/main/version3.x/paddleocr_and_paddlex.html#1-paddleocr-%E4%B8%8E-paddlex-%E7%9A%84%E5%8C%BA%E5%88%AB%E4%B8%8E%E8%81%94%E7%B3%BB) 了解 PaddleOCR 与 PaddleX 的关系。

## 4. 配置详解

你可以通过环境变量来控制服务器的行为。

-   `PADDLEOCR_MCP_PIPELINE`:
    -   **描述**: 选择要运行的产线。
    -   **可选值**: `"OCR"` (默认), `"PP-StructureV3"`。

-   `PADDLEOCR_MCP_PPOCR_SOURCE`:
    -   **描述**: OCR 功能的来源。
    -   **可选值**:
        -   `"local"` (默认): 直接在本地运行 PaddleOCR 模型。需要完整的本地依赖。
        -   `"aistudio"`: 通过 API 调用 AI Studio 的服务。
        -   `"self_hosted"`: 调用你自己部署的 PaddleOCR 服务。

-   `PADDLEOCR_MCP_SERVER_URL`:
    -   **描述**: 在 `aistudio` 或 `self_hosted` 模式下，指定后端服务的 URL。

-   `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`:
    -   **描述**: 在 `aistudio` 模式下，指定认证所需的 Access Token。

-   `PADDLEOCR_MCP_TIMEOUT`:
    -   **描述**: API 请求的超时时间（秒）。
    -   **默认值**: `30`。

## 许可证

[MIT License](LICENSE)
