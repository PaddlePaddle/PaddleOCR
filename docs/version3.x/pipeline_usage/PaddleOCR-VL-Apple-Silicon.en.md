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

> TIP:
> Before reading this hardware-specific tutorial, we recommend first reading the [Process Guide](./PaddleOCR-VL.en.md#process-guide) in the main [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md) to determine which chapters apply to your goal, and then returning here to read the corresponding sections.

## 1. Environment Preparation

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

Currently, only **manual deployment** is supported. Please refer to **Section 4.2 Method 2: Manual Deployment** in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.3 Client Invocation Methods

Please refer to [PaddleOCR-VL Usage Tutorial - 4.3 Client Invocation Methods](./PaddleOCR-VL.en.md#43-client-side-invocation).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to [PaddleOCR-VL Usage Tutorial - 4.4 Pipeline Configuration Adjustment Instructions](./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions).

## 5. Model Fine-Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 5. Model Fine-Tuning](./PaddleOCR-VL.en.md#5-model-fine-tuning).
