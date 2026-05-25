---
comments: true
---

# PaddleOCR-VL Apple Silicon Usage Tutorial

> INFO:
> Unless otherwise specified, the term "PaddleOCR-VL" in this tutorial refers to the PaddleOCR-VL model series (e.g., PaddleOCR-VL-1.5). References specific to the PaddleOCR-VL v1 version will be explicitly noted.

This tutorial is a guide for using PaddleOCR-VL on Apple Silicon, covering the complete workflow from environment preparation to service deployment.

Apple Silicon include, but are not limited to:

- Apple M1
- Apple M2
- Apple M3
- Apple M4

PaddleOCR-VL has been verified for accuracy and speed on the Apple M4. However, due to hardware diversity, compatibility with other Apple Silicon has not yet been confirmed. We welcome the community to test on different hardware setups and share your results.

## Workflow Guide for This Hardware

Use this guide for the workflows below.

| Goal | Support on this hardware | Read this section |
| --- | --- | --- |
| Local direct inference | Supported | Read Section 1. Environment Preparation and Section 2. Quick Start. |
| Client + VLM inference service | Supported | Complete local direct inference first, then read Section 3. Improving Inference Performance with VLM Inference Services. |
| Full API service | Supported with manual deployment only | Complete Section 1. Environment Preparation first, then read Section 4.1 Manual Deployment; after that, continue with Section 4.2 Client Invocation Methods and Section 4.3 Pipeline Configuration Adjustment Instructions. |
| Model fine-tuning | Supported | Read Section 5. Model Fine-Tuning. |

If you only need to confirm which inference methods are available on this hardware, refer to the [PaddleOCR-VL Inference Method and Hardware Support Matrix](./PaddleOCR-VL.en.md#inference-device-support-for-paddleocr-vl) in the main guide.

## 1. Environment Preparation

**Environment Setup Methods Supported on This Hardware**

| Environment setup method | Status | Notes |
| --- | --- | --- |
| Official Docker image | Not currently supported | This hardware does not currently support this path. |
| Manually install PaddlePaddle and PaddleOCR | Supported with steps in this guide | Continue reading this section. |

**We strongly recommend installing PaddleOCR-VL in a virtual environment to avoid dependency conflicts.** For example, use the Python venv standard library to create a virtual environment:

```shell
# Create a virtual environment
python -m venv .venv_paddleocr
# Activate the environment
source .venv_paddleocr/bin/activate
```

Execute the following commands to complete the installation:

```shell
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -U "paddleocr[doc-parser]"
```

> **Please install PaddlePaddle framework version 3.2.1 or above.**

## 2. Quick Start

Please refer to [PaddleOCR-VL Usage Tutorial - 2. Quick Start](./PaddleOCR-VL.en.md#2-quick-start).

## 3. Improving Inference Performance with VLM Inference Services

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This section introduces how to improve PaddleOCR-VL inference performance through a VLM inference service. In this hardware-specific guide, the examples use MLX-VLM as the backend for the VLM inference service.

### 3.1 Starting the VLM Inference Service

> IMPORTANT:
> The service started according to this section is responsible only for the VLM inference stage in the PaddleOCR-VL workflow. It does not provide a complete end-to-end document parsing API. We strongly recommend that you do not call this service directly via HTTP requests or OpenAI clients to process document images. If you need to deploy a service with the full PaddleOCR-VL capabilities, refer to the service deployment section later in this document.

**Launch Methods Supported on This Hardware**

| Launch method | Status | Notes |
| --- | --- | --- |
| Official Docker image | Not currently supported | This hardware does not currently support this path. |
| Install dependencies with the PaddleOCR CLI and launch the service | Not currently supported | This hardware does not currently support this path. |
| Launch the service directly with the acceleration framework | Supported with steps in this guide | This section provides the MLX-VLM launch steps. |

Install the MLX-VLM inference framework (v0.3.11 or later):

```shell
python -m pip install "mlx-vlm>=0.3.11"
```

Start the MLX-VLM inference service:

```shell
mlx_vlm.server --port 8111
```

### 3.2 Client Usage Method

The following invocation methods apply to an already launched MLX-VLM inference service.

#### 3.2.1 Command Line Usage

You can specify the backend type (`mlx-vlm-server`) via `--vl_rec_backend`, the service address via `--vl_rec_server_url`, and the huggingface repo id or server-side model weights path via `--vl_rec_api_model_name`. For example:

```shell
paddleocr doc_parser \
  --input paddleocr_vl_demo.png \
  --vl_rec_backend mlx-vlm-server \
  --vl_rec_server_url http://localhost:8111/ \
  --vl_rec_api_model_name PaddlePaddle/PaddleOCR-VL-1.5
```

#### 3.2.2 Python Script Integration

When creating a `PaddleOCRVL` object, specify the backend type via `vl_rec_backend`, the service address via the `vl_rec_server_url` parameter, and the huggingface repo id or server-side model weights path via `vl_rec_api_model_name`. For example:

```python
pipeline = PaddleOCRVL(
    vl_rec_backend="mlx-vlm-server", 
    vl_rec_server_url="http://localhost:8111/",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
)
```

### 3.3 Performance Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 3.3 Performance Tuning](./PaddleOCR-VL.en.md#33-performance-tuning).

## 4. Service Deployment

**Deployment Methods Supported on This Hardware**

| Deployment method | Status | Notes |
| --- | --- | --- |
| Docker Compose deployment | Not currently supported | This hardware currently supports only the manual deployment path. |
| Manual deployment | Supported; steps are in the main guide | Complete Section 1. Environment Preparation first, then continue with Section 4.1 and follow the shared manual deployment guidance in the main guide. |

### 4.1 Manual Deployment

Please complete Section 1. Environment Preparation first, then refer to [PaddleOCR-VL Usage Tutorial - 4.2 Method 2: Manual Deployment](./PaddleOCR-VL.en.md#42-method-2-manual-deployment).

### 4.2 Client Invocation Methods {#43-client-invocation-methods}

Please refer to [PaddleOCR-VL Usage Tutorial - 4.3 Client Invocation Methods](./PaddleOCR-VL.en.md#43-client-side-invocation).

### 4.3 Pipeline Configuration Adjustment Instructions {#44-pipeline-configuration-adjustment-instructions}

Please refer to [PaddleOCR-VL Usage Tutorial - 4.4 Pipeline Configuration Adjustment Instructions](./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions).

## 5. Model Fine-Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 5. Model Fine-Tuning](./PaddleOCR-VL.en.md#5-model-fine-tuning).
