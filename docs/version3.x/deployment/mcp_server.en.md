# PaddleOCR MCP Server

[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)

This project provides a lightweight [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) server designed to integrate the powerful capabilities of PaddleOCR into a compatible MCP Host.

### Key Features

- **Currently Supported Pipelines**
    - **OCR**: Performs text detection and recognition on images and PDF files.
    - **PP-StructureV3**: Recognizes and extracts text blocks, titles, paragraphs, images, tables, and other layout elements from an image or PDF file, converting the input into a Markdown document.
- **Supports the following working modes**:
    - **Local**: Runs the PaddleOCR pipeline directly on your machine using the installed Python library.
    - **AI Studio**: Calls cloud services provided by the Paddle AI Studio community.
    - **Self-hosted**: Calls a PaddleOCR service that you deploy yourself (serving).

### Table of Contents

- [1. Installation](#1-installation)
- [2. Quick Start](#2-quick-start)
- [3. Configuration](#3-configuration)
  - [3.1. MCP Host Configuration](#31-mcp-host-configuration)
  - [3.2. Working Modes Explained](#32-working-modes-explained)
    - [Mode 1: AI Studio Service (`aistudio`)](#mode-1-ai-studio-service-aistudio)
    - [Mode 2: Local Python Library (`local`)](#mode-2-local-python-library-local)
    - [Mode 3: Self-hosted Service (`self_hosted`)](#mode-3-self-hosted-service-self_hosted)
- [4. Parameter Reference](#4-parameter-reference)
- [5. Configuration Examples](#5-configuration-examples)
  - [5.1 AI Studio Service Configuration](#51-ai-studio-service-configuration)
  - [5.2 Local Python Library Configuration](#52-local-python-library-configuration)
  - [5.3 Self-hosted Service Configuration](#53-self-hosted-service-configuration)
- [6. Known Limitations](#6-known-limitations)

## 1. Installation

```bash
# Install the wheel
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.1.0/paddleocr_mcp-0.1.0-py3-none-any.whl

# Or, install from source
# git clone https://github.com/PaddlePaddle/PaddleOCR.git
# pip install -e mcp_server
```

You can check whether the installation was successful by running the following command:

```bash
paddleocr_mcp --help
```

If running the above command prints the help information, the installation was successful. This project depends on the `python-magic` library. If you see an error like the following when executing the command:

```
...
ImportError: failed to find libmagic.  Check your installation
```

it is likely due to missing underlying libraries required by `python-magic`. Please refer to the [official python-magic documentation](https://github.com/ahupp/python-magic?tab=readme-ov-file#installation) to install the necessary dependencies.

In addition, some [working modes](#32-working-modes-explained) may require extra dependencies. For more details, please refer to the following sections.

## 2. Quick Start

This section guides you through a quick setup using **Claude for Desktop** as the MCP Host and the **Local Python Library** mode. Please refer to [3. Configuration](#3-configuration) for other working modes and more configuration options.

1. **Prepare the Local Python Library PaddleOCR**

    - Refer to the [PaddleOCR Installation Guide](../installation.en.md) to install the PaddlePaddle framework and PaddleOCR. **It is strongly recommended to install them in a separate virtual environment** to avoid dependency conflicts.

2. **Locate the MCP Configuration File** - For details, refer to the [Official MCP Documentation](https://modelcontextprotocol.io/quickstart/user).

    - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    - **Linux**: `~/.config/Claude/claude_desktop_config.json`

3. **Add MCP Server Configuration**

    Open the `claude_desktop_config.json` file and add the configuration by referring to [5.2 Local Python Library Configuration](#52-local-python-library-configuration)

    **Note**:

    - If `paddleocr_mcp` is not in your system's `PATH`, set `command` to the absolute path of the executable.

4. **Restart the MCP Host**

    Restart Claude for Desktop. The new `paddleocr-ocr` tool should now be available in the application.

## 3. Configuration

### 3.1. MCP Host Configuration

In the Host's configuration file (e.g., `claude_desktop_config.json`), you need to define how to start the tool server. Key fields are:

- `command`: `paddleocr_mcp` (if the executable is in your `PATH`) or an absolute path.
- `args`: Configurable command-line arguments, e.g., `["--verbose"]`. See [4. Parameter Reference](#4-parameter-reference).
- `env`: Configurable environment variables. See [4. Parameter Reference](#4-parameter-reference).

### 3.2. Working Modes Explained

You can configure the MCP server to run in different modes based on your needs.

#### Mode 1: AI Studio Service (`aistudio`)
This mode calls services from the [Paddle AI Studio community](https://aistudio.baidu.com/pipeline/mine).

- **Use Case**: Ideal for quickly trying out features, validating solutions, and for no-code development scenarios.
- **Procedure**:

  1. Visit the [Paddle AI Studio community](https://aistudio.baidu.com/pipeline/mine) and log in. **Please note that AI Studio currently requires users to bind a mainland China phone number.** If you do not meet this requirement, please consider using an alternative working mode.
  2. In the "PaddleX Pipeline" section under "More" on the left, navigate to [Create Pipeline] - [OCR] - [General OCR] - [Deploy Directly] - [Text Recognition Module, select PP-OCRv5_server_rec] - [Start Deployment].
  3. Once deployed, obtain your **Service Base URL** (e.g., `https://xxxxxx.aistudio-hub.baidu.com`).
  4. Get your **Access Token** from [this page](https://aistudio.baidu.com/index/accessToken).

- In addition to using the platform's preset model solutions, you can also train and deploy custom models on the platform.

#### Mode 2: Local Python Library (`local`)
This mode runs the model directly on your local machine and has certain requirements for the local environment and computer performance. It relies on the installed `paddleocr` inference package.

- **Use Case**: Suitable for offline usage and scenarios with strict data privacy requirements.
- **Procedure**: Please refer to [2. Quick Start](#2-quick-start).

#### Mode 3: Self-hosted Service (`self_hosted`)
This mode calls a PaddleOCR inference service that you have deployed yourself. This corresponds to the **Serving** solutions provided by PaddleX.

- **Use Case**: Offers the advantages of serving and high flexibility, making it well-suited for scenarios requiring custom service configurations.
- **Procedure**:

    1. Refer to the [PaddleOCR Installation Guide](../installation.en.md) to install the PaddlePaddle framework and PaddleOCR.
    2. Refer to the [PaddleOCR Serving Guide](./serving.en.md) to run the server. Currently, only the basic serving solution is supported.
    3. Refer to [5.3 Self-hosted Service Configuration](#53-self-hosted-service-configuration) to modify the `claude_desktop_config.json` file.
    4. Set your service address in `PADDLEOCR_MCP_SERVER_URL` (e.g., `"http://127.0.0.1:8080"`).

## 4. Parameter Reference

You can control the server's behavior via environment variables or command-line arguments.

| Environment Variable | Command-line Argument | Type | Description | Options | Default |
|:---|:---|:---|:---|:---|:---|
| `PADDLEOCR_MCP_PIPELINE` | `--pipeline` | `str` | The pipeline to run | `"OCR"`, `"PP-StructureV3"` | `"OCR"` |
| `PADDLEOCR_MCP_PPOCR_SOURCE` | `--ppocr_source` | `str` | The source of PaddleOCR capabilities | `"local"`, `"aistudio"`, `"self_hosted"` | `"local"` |
| `PADDLEOCR_MCP_SERVER_URL` | `--server_url` | `str` | Base URL of the underlying service (required for `aistudio` or `self_hosted` mode) | - | `None` |
| `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN` | `--aistudio_access_token` | `str` | AI Studio authentication token (required for `aistudio` mode) | - | `None` |
| `PADDLEOCR_MCP_TIMEOUT` | `--timeout` | `int` | HTTP read timeout in seconds for API requests to the underlying server. | - | `60` |
| `PADDLEOCR_MCP_DEVICE` | `--device` | `str` | Specify the device for inference (only effective in `local` mode) | - | `None` |
| `PADDLEOCR_MCP_PIPELINE_CONFIG` | `--pipeline_config` | `str` | Path to the PaddleX pipeline configuration file (only effective in `local` mode) | - | `None` |
| - | `--http` | `bool` | Use Streamable HTTP transport instead of stdio (for remote deployment and multiple clients) | - | `False` |
| - | `--host` | `str` | Host address for Streamable HTTP mode | - | `"127.0.0.1"` |
| - | `--port` | `int` | Port for Streamable HTTP mode | - | `8080` |
| - | `--verbose` | `bool` | Enable verbose logging for debugging | - | `False` |

## 5. Configuration Examples

Below are complete Claude for Desktop configuration examples for different working modes. You can copy and modify them as needed.

### 5.1 AI Studio Service Configuration

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

**Note**:

- Replace `<your-server-url>` with your AI Studio **Service Base URL**, e.g., `https://xxxxx.aistudio-hub.baidu.com`. Do not include endpoint paths (like `/ocr`).
- Replace `<your-access-token>` with your **Access Token**.
- Do not leak your **Access Token**.

### 5.2 Local Python Library Configuration

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

**Note**:

- `PADDLEOCR_MCP_PIPELINE_CONFIG` is optional. If not set, the default pipeline configuration is used. To adjust settings, such as changing models, refer to the [PaddleOCR and PaddleX documentation](../paddleocr_and_paddlex.en.md), export a pipeline configuration file, and set `PADDLEOCR_MCP_PIPELINE_CONFIG` to its absolute path.
- **CPU Inference Performance Tip**:
  - **OCR Pipeline**: If you are running in a CPU environment, it is recommended to switch to the `mobile` series models for better performance. You can change the detection and recognition models in your pipeline configuration file to `text_detection_model_name="PP-OCRv5_mobile_det"` and `text_recognition_model_name="PP-OCRv5_mobile_rec"` respectively.
  - **PP-StructureV3 Pipeline**: Due to its model complexity, using this pipeline in an environment without a GPU is not recommended.

### 5.3 Self-hosted Service Configuration

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

**Note**:

- Replace `<your-server-url>` with the base URL of your underlying service (e.g., `http://127.0.0.1:8080`). 

## 6. Known Limitations

1. In `local` mode, the currently provided tools cannot handle Base64-encoded PDF document input.
2. In `local` mode, the currently provided tools do not infer the file type based on the model-specified `file_type` and may fail to process some complex URLs.
3. For the PP-StructureV3 pipeline, if the input file contains images, the returned result may significantly increase token usage. If image content is not needed, you can explicitly exclude it through the prompt to reduce resource consumption.
