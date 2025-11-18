---
comments: true
---

# PaddleOCR-VL-RTX50 Environment Configuration Tutorial

This tutorial is an environment configuration guide for NVIDIA RTX 50 series GPUs, aiming to complete the relevant environment setup. After completing the environment configuration, please refer to the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md) to use PaddleOCR-VL.

Before starting the tutorial, **please confirm that your NVIDIA driver supports CUDA 12.9 or later**.

## 1. Environment Preparation

This section mainly introduces how to set up the runtime environment for PaddleOCR-VL. There are two methods below; choose either one:

- Method 1: Use the official Docker image (not currently supported, adaptation in progress).

- Method 2: Manually install PaddlePaddle and PaddleOCR.

### 1.1 Method 1: Using Docker Image

Not currently supported, adaptation in progress.

### 1.2 Method 2: Manually Install PaddlePaddle and PaddleOCR

If you cannot use Docker, you can also manually install PaddlePaddle and PaddleOCR. Python version 3.8–3.12 is required.

**We strongly recommend installing PaddleOCR-VL in a virtual environment to avoid dependency conflicts.** For example, use the Python `venv` standard library to create a virtual environment:

```shell
# Create a virtual environment
python -m venv .venv_paddleocr
# Activate the environment
source .venvenv_paddleocr/bin/activate
```

Run the following commands to complete the installation:

```shell
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install -U "paddleocr[doc-parser]"
# For Linux systems, run:
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
# For Windows systems, run:
python -m pip install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl
```

> **Please ensure that you install PaddlePaddle version 3.2.1 or later, along with the special version of safetensors.**

## 2. Quick Start

Please refer to the same section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 3. Improving VLM Inference Performance Using Inference Acceleration Frameworks

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This section mainly introduces how to use the vLLM and SGLang inference acceleration frameworks to improve the inference performance of PaddleOCR-VL.

### 3.1 Starting the VLM Inference Service

There are two methods to start the VLM inference service; choose either one:

- Method 1: Use the official Docker image to start the service (not currently supported, adaptation in progress).

- Method 2: Manually install dependencies via the PaddleOCR CLI and then start the service.

#### 3.1.1 Method 1: Using Docker Image

Not currently supported, adaptation in progress.

#### 3.1.2 Method 2: Installation and Usage via PaddleOCR CLI

Since inference acceleration frameworks may have dependency conflicts with PaddlePaddle, it is recommended to install them in a virtual environment. Taking vLLM as an example:

```shell
# If there is an currently activated virtual environment, deactivate it first using `deactivate`
# Create a virtual environment
python -m venv .venv_vlm
# Activate the environment
source .venv_vlm/bin/activate
# Install PaddleOCR
python -m pip install "paddleocr[doc-parser]"
# Install dependencies for the inference acceleration service
paddleocr install_genai_server_deps vllm
python -m pip install flash-attn==2.8.3
```

> The `paddleocr install_genai_server_deps` command may require CUDA compilation tools such as `nvcc` during execution. If these tools are not available in your environment or the installation takes too long, you can obtain a precompiled version of FlashAttention from [this repository](https://github.com/mjun0812/flash-attention-prebuild-wheels), for example, by running `python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl`.

Usage of the `paddleocr install_genai_server_deps` command:

```shell
paddleocr install_genai_server_deps <inference acceleration framework name>
```

The currently supported framework names are `vllm` and `sglang`, corresponding to vLLM and SGLang, respectively.

After installation, you can start the service using the `paddleocr genai_server` command:

```shell
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

The supported parameters for this command are as follows:

| Parameter          | Description                          |
|-------------------|--------------------------------------|
| `--model_name`    | Model name                           |
| `--model_dir`     | Model directory                       |
| `--host`          | Server hostname                       |
| `--port`          | Server port number                    |
| `--backend`       | Backend name, i.e., the name of the inference acceleration framework used; options are `vllm` or `sglang` |
| `--backend_config` | Specify a YAML file containing backend configurations |

### 3.2 Client Usage

Please refer to the same section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).## 4. Service-Oriented Deployment

This section primarily introduces how to deploy PaddleOCR-VL as a service and invoke it. There are two methods available; choose either one:

- Method 1: Deployment using Docker Compose (currently not supported, adaptation in progress).

- Method 2: Manual installation of dependencies for deployment.

Please note that the PaddleOCR-VL service introduced in this section differs from the VLM inference service in the previous section: the latter is responsible for only one part of the complete workflow (i.e., VLM inference) and is called as an underlying service by the former.

### 4.1 Method 1: Deployment Using Docker Compose

Currently not supported, adaptation in progress.

### 4.2 Method 2: Manual Installation of Dependencies for Deployment

Execute the following commands to install the service deployment plugin via the PaddleX CLI:

```shell
paddlex --install serving
```

Then, start the server using the PaddleX CLI:

```shell
paddlex --serve --pipeline PaddleOCR-VL
```

After startup, you will see output similar to the following. The server listens on port **8080** by default:

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

The command-line parameters related to service-oriented deployment are as follows:

<table>
<thead>
<tr>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>Registered name of the PaddleX pipeline or path to the pipeline configuration file.</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>Device for pipeline deployment. By default, the GPU is used if available; otherwise, the CPU is used.</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>Hostname or IP address to which the server is bound. The default is <code>0.0.0.0</code>.</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>Port number on which the server listens. The default is <code>8080</code>.</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>Enable high-performance inference mode. Refer to the high-performance inference documentation for more information.</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>High-performance inference configuration. Refer to the high-performance inference documentation for more information.</td>
</tr>
</tbody>
</table>

To adjust pipeline-related configurations (such as model paths, batch sizes, deployment devices, etc.), refer to Section 4.4.

### 4.3 Client Invocation Method

Refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.4 Pipeline Configuration Adjustment Instructions

Refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 5. Model Fine-Tuning

Refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).
