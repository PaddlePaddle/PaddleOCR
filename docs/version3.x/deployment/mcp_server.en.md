---
comments: true
---

# PaddleOCR MCP Server

[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![FastMCP](https://img.shields.io/badge/Built%20with-FastMCP%20v2-blue)](https://gofastmcp.com)

This project provides a lightweight [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) server designed to integrate PaddleOCR capabilities into various LLM applications.

### Key Features

- **Currently Supported Tools**
    - **OCR**: Performs text detection and recognition on images and PDF files.
    - **PP-StructureV3**: Identifies and extracts text blocks, titles, paragraphs, images, tables, and other layout elements from images or PDF files, converting the input into Markdown documents.
- **Supported Working Modes**
    - **Local Python Library**: Runs PaddleOCR pipelines directly on the local machine. This mode requires a suitable local environment and hardware, and is ideal for offline use or privacy-sensitive scenarios.
    - **AI Studio Community Service**: Invokes services hosted on the [PaddlePaddle AI Studio Community](https://aistudio.baidu.com/pipeline/mine). This is suitable for quick testing, prototyping, or no-code scenarios.
    - **Self-hosted Service**: Invokes the user's self-hosted PaddleOCR services. This mode offers the advantages of serving and high flexibility. It is suitable for scenarios requiring customized service configurations, as well as those with strict data privacy requirements. **Currently, only the basic serving solution is supported.**

## Examples:
The following showcases creative use cases built with PaddleOCR MCP server combined with other tools:

### Demo 1
In Claude for Desktop, extract handwritten content from images and save to note-taking software Notion. The PaddleOCR MCP server extracts text, formulas and other information from images while preserving document structure.
<div align="center">
  <img width="65%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/mcp_demo/note_to_notion.gif" alt="note_to_notion">
  <img width="30%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/mcp_demo/note.jpg" alt="note">
</div>

- Note: In addition to the PaddleOCR MCP server, this demo also uses the [Notion MCP server](https://developers.notion.com/docs/mcp).

---

### Demo 2
In VSCode, convert handwritten ideas or pseudocode into runnable Python scripts that comply with project coding standards with one click, and upload them to GitHub repositories. The PaddleOCR MCP server extracts explicitly handwritten code from images for subsequent processing.

<div align="center">
  <img width="70%" img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/mcp_demo/code_to_github.gif" alt="code_to_github">
</div>

- In addition to the PaddleOCR MCP server, this demo also uses the [filesystem MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem).

---

### Demo 3
In Claude for Desktop, convert PDF documents or images containing complex tables, formulas, handwritten text and other content into locally editable files.

#### Demo 3.1
Convert complex PDF documents with tables and watermarks to editable doc/Word format:
<div align="center">
  <img width="70%" img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/mcp_demo/pdf_to_file.gif" alt="pdf_to_file">
</div>

#### Demo 3.2
Convert images containing formulas and tables to editable csv/Excel format:
<div align="center">
  <img width="70%" img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/00136903a4d0b5f11bd978cb0ef5d3c44f3aa5e9/images/paddleocr/mcp_demo/table_to_excel1.png" alt="table_to_excel1">
  <img width="50%" img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/00136903a4d0b5f11bd978cb0ef5d3c44f3aa5e9/images/paddleocr/mcp_demo/table_to_excel2.png" alt="table_to_excel2">
  <img width="45%" img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/00136903a4d0b5f11bd978cb0ef5d3c44f3aa5e9/images/paddleocr/mcp_demo/table_to_excel3.png" alt="table_to_excel3">
</div>

- In addition to the PaddleOCR MCP server, this demo also uses the [filesystem MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem).

---

### Table of Contents

- [Table of Contents](#table-of-contents)
- [1. Installation](#1-installation)
- [2. Using with Claude for Desktop](#2-using-with-claude-for-desktop)
    - [2.1 Quick Start](#21-quick-start)
    - [2.2 MCP Host Configuration Details](#22-mcp-host-configuration-details)
    - [2.3 Working Modes Explained](#23-working-modes-explained)
    - [2.4 Using `uvx`](#24-using-uvx)
- [3. Running the Server](#3-running-the-server)
- [4. Parameter Reference](#4-parameter-reference)
- [5. Known Limitations](#5-known-limitations)

## 1. Installation

This section explains how to install the `paddleocr-mcp` library via pip.

- For the local Python library mode, you need to install both `paddleocr-mcp` and the PaddlePaddle framework along with PaddleOCR, as per the [PaddleOCR installation documentation](../installation.en.md).
- For the AI Studio community service or the self-hosted service modes, if used within MCP hosts like Claude for Desktop, the server can also be run without installation via tools like `uvx`. See [2. Using with Claude for Desktop](#2-using-with-claude-for-desktop) for details.

For the local Python library mode you may optionally choose convenience extras (helpful to reduce manual dependency steps):

- `paddleocr-mcp[local]`: Includes PaddleOCR (but NOT the PaddlePaddle framework itself).
- `paddleocr-mcp[local-cpu]`: Includes PaddleOCR AND the CPU version of the PaddlePaddle framework.

It is still recommended to use an isolated virtual environment to avoid conflicts.

To install `paddleocr-mcp` using pip:

```bash
# Install the wheel
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.2.1/paddleocr_mcp-0.2.1-py3-none-any.whl

# Or install from source
# git clone https://github.com/PaddlePaddle/PaddleOCR.git
# pip install -e mcp_server

# Install with optional extras (choose ONE of the following if you prefer convenience installs)
# Install PaddleOCR together with the MCP server (framework not included):
pip install "paddleocr-mcp[local] @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.2.1/paddleocr_mcp-0.2.1-py3-none-any.whl"

# Install PaddleOCR and CPU PaddlePaddle framework together:
pip install "paddleocr-mcp[local-cpu] @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.2.1/paddleocr_mcp-0.2.1-py3-none-any.whl"
```

To verify successful installation:

```bash
paddleocr_mcp --help
```

If the help message is printed, the installation succeeded.

## 2. Using with Claude for Desktop

This section explains how to use the PaddleOCR MCP server within Claude for Desktop. The steps are also applicable to other MCP hosts with minor adjustments.

### 2.1 Quick Start

1. **Install `paddleocr-mcp`**

    Refer to [1. Installation](#1-installation). To avoid dependency conflicts, **it is strongly recommended to install in an isolated virtual environment**.

2. **Install PaddleOCR**

    Install the PaddlePaddle framework and PaddleOCR, as per the [PaddleOCR installation documentation](../installation.en.md).

3. **Add MCP Server Configuration**

    Locate the `claude_desktop_config.json` configuration file:

    - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    - **Linux**: `~/.config/Claude/claude_desktop_config.json`

    Edit the file as follows:

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

    **Notes**:

    - `PADDLEOCR_MCP_PIPELINE_CONFIG` is optional; if not set, the default pipeline configuration will be used. If you need to adjust the configuration, such as changing the model, please refer to the [PaddleOCR documentation](../paddleocr_and_paddlex.md) to export the pipeline configuration file, and set `PADDLEOCR_MCP_PIPELINE_CONFIG` to the absolute path of this configuration file.

    - **Inference Performance Tips**:

        If you encounter issues such as long inference time or insufficient memory during use, you may consider adjusting the pipeline configuration according to the following recommendations.

        - **OCR Pipeline**: It is recommended to switch to the `mobile` series models. For example, you can modify the pipeline configuration file to use `PP-OCRv5_mobile_det` for detection and `PP-OCRv5_mobile_rec` for recognition.

        - **PP-StructureV3 Pipeline**:

            - Disable unused features, e.g., set `use_formula_recognition` to `False` to disable formula recognition.
            - Use lightweight models, such as replacing the OCR model with the `mobile` version or switching to a lightweight formula recognition model like PP-FormulaNet-S.

            The following sample code can be used to obtain the pipeline configuration file, in which most optional features of the PP-StructureV3 pipeline are disabled, and some key models are replaced with lightweight versions.

            ```python
            from paddleocr import PPStructureV3

            pipeline = PPStructureV3(
                use_doc_orientation_classify=False, # Disable document image orientation classification
                use_doc_unwarping=False,            # Disable text image unwarping
                use_textline_orientation=False,     # Disable text line orientation classification
                use_formula_recognition=False,      # Disable formula recognition
                use_seal_recognition=False,         # Disable seal text recognition
                use_table_recognition=False,        # Disable table recognition
                use_chart_recognition=False,        # Disable chart parsing
                # Use lightweight models
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                layout_detection_model_name="PP-DocLayout-S",
            )

            # The configuration file is saved to `PP-StructureV3.yaml`
            pipeline.export_paddlex_config_to_yaml("PP-StructureV3.yaml")
            ```

      **Important**:

      - If `paddleocr_mcp` is not in your system's `PATH`, set `command` to the absolute path of the executable.

4. **Restart the MCP Host**

    Restart Claude for Desktop. The `paddleocr-ocr` tool should now be available in the application.

### 2.2 MCP Host Configuration Details

In the configuration file for Claude for Desktop, you need to define how the MCP server is started. The key fields are as follows:

- `command`: `paddleocr_mcp` (if the executable can be found in the `PATH`) or the absolute path.
- `args`: Configurable command-line arguments, such as `["--verbose"]`. See [4. Parameter Reference](#4-parameter-reference) for details.
- `env`: Configurable environment variables. See [4. Parameter Reference](#4-parameter-reference) for details.

### 2.3 Working Modes Explained

You can configure the MCP server according to your requirements to run in different working modes. The operational procedures vary for different modes, which will be explained in detail below.

#### Mode 1: Local Python Library

See [2.1 Quick Start](#21-quick-start).

#### Mode 2: AI Studio Community Service

1. Install `paddleocr-mcp`.
2. Set up AI Studio community service.
    - Visit [PaddlePaddle AI Studio Community](https://aistudio.baidu.com/pipeline/mine) and log in.
    - Under "PaddleX Pipeline" in the "More" section on the left, click in sequence: [Create Pipeline] - [OCR] - [General OCR] - [Deploy Directly] - [Start Deployment].
    - After deployment, obtain your **service base URL** (e.g., `https://xxxxxx.aistudio-hub.baidu.com`).
    - Get your **access token** from [this page](https://aistudio.baidu.com/index/accessToken).
3. Refer to the configuration example below to modify the contents of the `claude_desktop_config.json` file.
3. Restart the MCP host.

Configuration example:

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

**Notes**:

- Replace `<your-server-url>` with your AI Studio service base URL, e.g., `https://xxxxx.aistudio-hub.baidu.com`. Make sure not to include the endpoint path (such as `/ocr`).
- Replace `<your-access-token>` with your access token.

**Important**:

- Do not expose your access token.

You may also train and deploy custom models on the platform.

#### Mode 3: Self-hosted Service

1. In the environment where you need to run the PaddleOCR inference server, run the inference server as per the [PaddleOCR serving documentation](./serving.en.md).
2. Install `paddleocr-mcp` where the MCP server will run.
3. Refer to the configuration example below to modify the contents of the `claude_desktop_config.json` file.
4. Set `PADDLEOCR_MCP_SERVER_URL` (e.g., `"http://127.0.0.1:8000"`).
5. Restart the MCP host.

Configuration example:

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

- Replace `<your-server-url>` with your service’s base URL (e.g., `http://127.0.0.1:8000`).

### 2.4 Using `uvx`

Currently, for the AI Studio and self-hosted modes, and (for CPU inference) the local mode, starting the MCP server via `uvx` is also supported. With this approach, manual installation of `paddleocr-mcp` is not required. The main steps are as follows:

1. Install [uv](https://docs.astral.sh/uv/#installation).
2. Modify `claude_desktop_config.json`. Examples:

  Self-hosted mode:

    ```json
    {
      "mcpServers": {
        "paddleocr-ocr": {
          "command": "uvx",
          "args": [
            "--from",
            "paddleocr-mcp@https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.2.1/paddleocr_mcp-0.2.1-py3-none-any.whl",
            "paddleocr_mcp"
          ],
          "env": {
            "PADDLEOCR_MCP_PIPELINE": "OCR",
            "PADDLEOCR_MCP_PPOCR_SOURCE": "self_hosted",
            "PADDLEOCR_MCP_SERVER_URL": "<your-server-url>"
          }
        }
      }
    }
    ```

    Local mode (CPU, using the `local-cpu` extra):

    ```json
    {
      "mcpServers": {
        "paddleocr-ocr": {
          "command": "uvx",
          "args": [
            "--from",
            "paddleocr_mcp[local-cpu]@https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/mcp/paddleocr_mcp/releases/v0.2.1/paddleocr_mcp-0.2.1-py3-none-any.whl",
            "paddleocr_mcp"
          ],
          "env": {
            "PADDLEOCR_MCP_PIPELINE": "OCR",
            "PADDLEOCR_MCP_PPOCR_SOURCE": "local"
          }
        }
      }
    }
    ```

    Because a different startup method is used (`uvx` pulls and runs the package on-demand), only `command` and `args` differ from earlier examples; available environment variables and CLI arguments remain identical.

## 3. Running the Server

In addition to MCP hosts like Claude for Desktop, you can also run the PaddleOCR MCP server via the CLI.

To view help:

```bash
paddleocr_mcp --help
```

Example commands:

```bash
# OCR + AI Studio community service + stdio
PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN=xxxxxx paddleocr_mcp --pipeline OCR --ppocr_source aistudio --server_url https://xxxxxx.aistudio-hub.baidu.com

# PP-StructureV3 + local Python library + stdio
paddleocr_mcp --pipeline PP-StructureV3 --ppocr_source local

# OCR + self-hosted service + Streamable HTTP
paddleocr_mcp --pipeline OCR --ppocr_source self_hosted --server_url http://127.0.0.1:8080 --http
```

You can find all the supported parameters of the PaddleOCR MCP server in [4. Parameter Reference](#4-parameter-reference).

## 4. Parameter Reference

You can control the MCP server via environment variables or CLI arguments.

| Environment Variable                          | CLI Argument              | Type   | Description                                                           | Options                                  | Default       |
| ------------------------------------- | ------------------------- | ------ | --------------------------------------------------------------------- | ---------------------------------------- | ------------- |
| `PADDLEOCR_MCP_PIPELINE`              | `--pipeline`              | `str`  | Pipeline to run.                                                      | `"OCR"`, `"PP-StructureV3"`              | `"OCR"`       |
| `PADDLEOCR_MCP_PPOCR_SOURCE`          | `--ppocr_source`          | `str`  | Source of PaddleOCR capabilities.                                     | `"local"` (local Python library), `"aistudio"` (AI Studio community service), `"self_hosted"` (self-hosted service) | `"local"`     |
| `PADDLEOCR_MCP_SERVER_URL`            | `--server_url`            | `str`  | Base URL for the underlying service (`aistudio` or `self_hosted` mode only). | -                                        | `None`        |
| `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN` | `--aistudio_access_token` | `str`  | AI Studio access token (`aistudio` mode only).                 | -                                        | `None`        |
| `PADDLEOCR_MCP_TIMEOUT`               | `--timeout`               | `int`  | Read timeout for the underlying requests (seconds).                          | -                                        | `60`          |
| `PADDLEOCR_MCP_DEVICE`                | `--device`                | `str`  | Device for inference (`local` mode only).                          | -                                        | `None`        |
| `PADDLEOCR_MCP_PIPELINE_CONFIG`       | `--pipeline_config`       | `str`  | Path to pipeline config file (`local` mode only).                     | -                                        | `None`        |
| -                                     | `--http`                  | `bool` | Use Streamable HTTP instead of stdio (for remote/multi-client use).   | -                                        | `False`       |
| -                                     | `--host`                  | `str`  | Host for the Stremable HTTP mode.                                                   | -                                        | `"127.0.0.1"` |
| -                                     | `--port`                  | `int`  | Port for the Streamable HTTP mode.                                                   | -                                        | `8000`        |
| -                                     | `--verbose`               | `bool` | Enable verbose logging for debugging.                                               | -                                        | `False`       |

## 5. Known Limitations

- In the local Python library mode, the current tools cannot process PDF document inputs that are Base64 encoded.
- In the local Python library mode, the current tools do not infer the file type based on the model's `file_type` prompt, and may fail to process some complex URLs.
- For the PP-StructureV3 pipeline, if the input file contains images, the returned results may significantly increase token usage. If image content is not needed, you can explicitly exclude it through prompts to reduce resource consumption.
