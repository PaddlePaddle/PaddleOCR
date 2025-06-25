# PaddleOCR MCP 服务器

[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)

本项目提供一个轻量级的 [Model Context Protocol（MCP）](https://modelcontextprotocol.io/introduction) 服务器，旨在将 PaddleOCR 的强大能力集成到兼容的 MCP Host 中。

### 主要功能

- **当前支持的工具**
    - **OCR**：对图像和 PDF 文件进行文本检测与识别。
    - **PP-StructureV3**：从图像或 PDF 文件中识别和提取文本块、标题、段落、图片、表格以及其他版面元素，将输入转换为 Markdown 文档。
- **支持运行在如下工作模式**
    - **本地 Python 库**：在本机直接运行 PaddleOCR 产线。
    - **星河社区服务**：调用飞桨星河社区提供的云端服务。
    - **自托管服务**：调用用户自行部署的 PaddleOCR 服务。

### 目录

- [1. 安装](#1-安装)
- [2. 快速开始](#2-快速开始)
- [3. 配置说明](#3-配置说明)
  - [3.1. MCP Host 配置](#31-mcp-host-配置)
  - [3.2. 工作模式详解](#32-工作模式详解)
    - [模式一：托管在星河社区的服务](#模式一托管在星河社区的服务-aistudio)
    - [模式二：本地 Python 库](#模式二本地-python-库-local)
    - [模式三：自托管服务](#模式三自托管服务-self_hosted)
- [4. 参数参考](#4-参数参考)
- [5. 配置示例](#5-配置示例)
  - [5.1 星河社区服务配置](#51-ai-studio-星河社区服务配置)
  - [5.2 本地 Python 库配置](#52-本地-python-库配置)
  - [5.3 自托管服务配置](#53-自托管服务配置)
- [6. 已知局限性](#6-已知局限性)

## 1. 安装

```bash
# 安装 wheel 包
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.1.0/paddleocr_mcp-0.1.0-py3-none-any.whl

# 或者，从项目源码安装
# git clone https://github.com/PaddlePaddle/PaddleOCR.git
# pip install -e mcp_server
```

可通过以下命令检查是否安装成功：

```bash
paddleocr_mcp --help
```

如果执行上述命令后打印出了帮助信息，则说明安装成功。本项目依赖 python-magic 库。如果在执行上述命令时出现如下错误提示：

```
...
ImportError: failed to find libmagic.  Check your installation
```

很可能是因为缺少 python-magic 库所需的底层库。请参考 [python-magic 官方文档](https://github.com/ahupp/python-magic?tab=readme-ov-file#installation) 完成相应依赖库的安装。

此外，部分 [工作模式](#32-工作模式详解) 可能需要安装额外依赖，详情请参考后续说明。

## 2. 快速开始

本节将以 **Claude for Desktop** 作为 MCP Host，并以 **星河社区服务** 工作模式为例，引导您完成快速配置。此模式无需在本地安装复杂的依赖，推荐新用户使用。请参考 [3. 配置说明](#3-配置说明) 了解其他工作模式的操作流程以及更多配置项。

1. **准备星河社区服务**

    - 访问 [飞桨星河社区](https://aistudio.baidu.com/pipeline/mine) 并登录。**请注意，目前星河社区要求用户绑定中国大陆手机号。** 如果您不具备此条件，请考虑使用其他工作模式。
    - 在左侧"更多内容"下的 "PaddleX 产线" 部分，[创建产线] - [OCR] - [通用 OCR] - [直接部署] - [文本识别模块，选择 PP-OCRv5_server_rec] - [开始部署]。
    - 部署成功后，获取您的 **服务基础 URL**（示例：`https://xxxxxx.aistudio-hub.baidu.com`）。
    - 在 [此页面](https://aistudio.baidu.com/index/accessToken) 获取您的 **访问令牌**。

2. **定位 MCP 配置文件** - 详情请参考 [MCP 官方文档](https://modelcontextprotocol.io/quickstart/user)。

    - **macOS**：`~/Library/Application Support/Claude/claude_desktop_config.json`
    - **Windows**：`%APPDATA%\Claude\claude_desktop_config.json`
    - **Linux**：`~/.config/Claude/claude_desktop_config.json`

3. **添加 MCP 服务器配置**

    打开 `claude_desktop_config.json` 文件，参考 [5.1 星河社区服务配置](#51-星河社区服务配置) 调整配置，填充到 `claude_desktop_config.json` 中。

    **注意**：

    - 请勿泄漏您的 **访问令牌**。
    - 如果 `paddleocr_mcp` 无法在系统 `PATH` 中找到，请将 `command` 设置为可执行文件的绝对路径。

4. **重启 MCP Host**

    重启 Claude for Desktop。新的 `paddleocr-ocr` 工具现在应该可以在应用中使用了。

## 3. 配置说明

### 3.1. MCP Host 配置

在 Host 的配置文件中（如 `claude_desktop_config.json`），您需要定义工具服务器的启动方式。关键字段如下：

- `command`：`paddleocr_mcp`（如果可执行文件可在 `PATH` 中找到）或绝对路径。
- `args`：可配置命令行参数，如 `["--verbose"]`。详见 [4. 参数参考](#4-参数参考)。
- `env`：可配置环境变量。详见 [4. 参数参考](#4-参数参考)。

### 3.2. 工作模式详解

您可以根据需求配置 MCP 服务器，使其运行在不同的工作模式。

#### 模式一：托管在星河社区的服务 (`aistudio`)
此模式调用 [飞桨星河社区](https://aistudio.baidu.com/pipeline/mine) 的服务。

- **适用场景**：适合快速体验功能、快速验证方案等，也适用于零代码开发场景。
- **操作流程**：请参考 [2. 快速开始](#2-快速开始)。
- 除了使用平台预设的模型方案，您也可以在平台上自行训练并部署自定义模型。

#### 模式二：本地 Python 库 (`local`)
此模式直接在本地计算机上运行模型，对本地环境与计算机性能有一定要求。

- **适用场景**：需要离线使用、对数据隐私有严格要求的场景。
- **操作流程**：

    1. 参考 [PaddleOCR 安装文档](../installation.md) 安装飞桨框架和 PaddleOCR。为避免依赖冲突，**强烈建议在独立的虚拟环境中安装**。
    2. 参考 [配置示例](#52-本地-python-库配置) 更改 `claude_desktop_config.json` 文件内容。

#### 模式三：自托管服务 (`self_hosted`)
此模式调用您自行部署的 PaddleOCR 推理服务。

- **适用场景**：具备服务化部署优势及高度灵活性，适用于需要自定义服务配置的场景。
- **操作流程**：

    1. 参考 [PaddleOCR 安装文档](../installation.md) 安装飞桨框架和 PaddleOCR。
    2. 参考 [PaddleOCR 服务化部署文档](./serving.md) 运行服务器。**目前暂时只支持基础服务化部署方案。**
    3. 参考 [配置示例](#53-自托管服务配置) 更改 `claude_desktop_config.json` 文件内容。
    4. 将您的服务地址填入 `PADDLEOCR_MCP_SERVER_URL` (例如：`"http://127.0.0.1:8000"`)。

## 4. 参数参考

您可以通过环境变量或命令行参数来控制服务器的行为。

| 环境变量 | 命令行参数 | 类型 | 描述 | 可选值 | 默认值 |
|:---------|:-----------|:-----|:-----|:-------|:-------|
| `PADDLEOCR_MCP_PIPELINE` | `--pipeline` | `str` | 要运行的产线 | `"OCR"`, `"PP-StructureV3"` | `"OCR"` |
| `PADDLEOCR_MCP_PPOCR_SOURCE` | `--ppocr_source` | `str` | PaddleOCR 能力来源 | `"local"`, `"aistudio"`, `"self_hosted"` | `"local"` |
| `PADDLEOCR_MCP_SERVER_URL` | `--server_url` | `str` | 底层服务基础 URL（`aistudio` 或 `self_hosted` 模式下必需） | - | `None` |
| `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN` | `--aistudio_access_token` | `str` | AI Studio 认证令牌（`aistudio` 模式下必需） | - | `None` |
| `PADDLEOCR_MCP_TIMEOUT` | `--timeout` | `int` | 底层服务请求的读取超时时间（秒） | - | `60` |
| `PADDLEOCR_MCP_DEVICE` | `--device` | `str` | 指定运行推理的设备（仅在 `local` 模式下生效） | - | `None` |
| `PADDLEOCR_MCP_PIPELINE_CONFIG` | `--pipeline_config` | `str` | PaddleOCR 产线配置文件路径（仅在 `local` 模式下生效） | - | `None` |
| - | `--http` | `bool` | 使用 Streamable HTTP 传输而非 stdio（适用于远程部署和多客户端） | - | `False` |
| - | `--host` | `str` | Streamable HTTP 模式的主机地址 | - | `"127.0.0.1"` |
| - | `--port` | `int` | Streamable HTTP 模式的端口 | - | `8000` |
| - | `--verbose` | `bool` | 启用详细日志记录，便于调试 | - | `False` |

## 5. 配置示例

以下是针对不同工作模式的 Claude for Desktop 完整配置示例，您可以直接复制并根据需要修改：

### 5.1 星河社区服务配置

```json
{
  "mcpServers": {
    "paddleocr-ocr": {
      "command": "paddleocr_mcp",
      "args": [],
      "env": {
        "PADDLEOCR_MCP_PIPELINE": "OCR",
        "PADDLEOCR_MCP_PPOCR_SOURCE": "aistudio",
        "PADDLEOCR_MCP_SERVER_URL": "<your-server-url>", 
        "PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN": "<your-access-token>"
      }
    }
  }
}
```

**说明**：

- 将 `<your-server-url>` 替换为您的星河社区服务的 **服务基础 URL**，例如 `https://xxxxx.aistudio-hub.baidu.com`，注意不要带有端点路径（如 `/ocr`）。
- 将 `<your-access-token>` 替换为您的 **访问令牌**。

### 5.2 本地 Python 库配置

```json
{
  "mcpServers": {
    "paddleocr-ocr": {
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

**说明**：

- `PADDLEOCR_MCP_PIPELINE_CONFIG` 为可选项，不设置时使用产线默认配置。如需调整配置，例如更换模型，请参考 [PaddleOCR 文档](../paddleocr_and_paddlex.md) 导出产线配置文件，并将 `PADDLEOCR_MCP_PIPELINE_CONFIG` 设置为配置文件的绝对路径。
- **CPU 推理性能提示**：
  - **OCR 产线**：默认使用的模型复杂度较高，如果您希望提升产线推理速度、降低内存消耗，建议更换 `mobile` 系列模型。例如，您可以在产线配置文件中将检测和识别模型分别修改为 `PP-OCRv5_mobile_det` 和 `PP-OCRv5_mobile_rec`。
  - **PP-StructureV3 产线**：使用默认配置需要消耗较多计算资源，如果您希望提升产线推理速度、降低内存消耗，请参考如下建议调整配置：
    
    - 关闭不需要用到的功能，例如设置 `use_formula_recognition` 为 `False` 以禁用公式识别。
    - 使用轻量级的模型，例如将 OCR 模型替换为 `mobile` 版本、换用轻量的公式识别模型 PP-FormulaNet-S 等。

    以下示例代码可用于获取产线配置文件，其中关闭了 PP-StructureV3 产线的大部分可选功能，同时将部分关键模型更换为轻量级版本。

    ```python
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        use_doc_orientation_classify=False, # 禁用文档图像方向分类
        use_doc_unwarping=False,            # 禁用文本图像矫正
        use_textline_orientation=False,     # 禁用文本行方向分类
        use_formula_recognition=False,      # 禁用公式识别
        use_seal_recognition=False,         # 禁用印章文本识别
        use_table_recognition=False,        # 禁用表格识别
        use_chart_recognition=False,        # 禁用图表解析
        # 使用轻量级模型
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        layout_detection_model_name="PP-DocLayout-S",
    )

    # 配置文件保存到 `PP-StructureV3.yaml` 中
    pipeline.export_paddlex_config_to_yaml("PP-StructureV3.yaml")
    ```

### 5.3 自托管服务配置

```json
{
  "mcpServers": {
    "paddleocr-ocr": {
      "command": "paddleocr_mcp",
      "args": [],
      "env": {
        "PADDLEOCR_MCP_PIPELINE": "OCR",
        "PADDLEOCR_MCP_PPOCR_SOURCE": "self_hosted",
        "PADDLEOCR_MCP_SERVER_URL": "<your-server-url>"
      }
    }
  }
}
```

**说明**：

- 将 `<your-server-url>` 替换为底层服务的基础 URL（如：`http://127.0.0.1:8000`）。

## 6. 已知局限性

1. 在 `local` 模式下，当前提供的工具无法处理 Base64 编码的 PDF 文档输入。
2. 在 `local` 模式下，当前提供的工具不会根据模型提示的 `file_type` 推断文件类型，对于一些复杂 URL 可能处理失败。
3. 对于 PP-StructureV3 产线，若输入文件中包含图像，返回结果可能会显著增加 token 使用量。若无需图像内容，可通过提示词明确排除，以降低资源消耗。
