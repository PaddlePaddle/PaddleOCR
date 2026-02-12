---
comments: true
---

# PaddleOCR-VL Apple Silicon Environment Configuration Tutorial

This tutorial is a guide for configuring the PaddleOCR-VL Apple Silicon environment. The purpose is to complete the relevant environment setup. After the environment configuration is complete, please refer to the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md) to use PaddleOCR-VL.

Apple Silicon include, but are not limited to:

- Apple M1
- Apple M2
- Apple M3
- Apple M4

PaddleOCR-VL has been verified for accuracy and speed on the Apple M4. However, due to hardware diversity, compatibility with other Apple Silicon has not yet been confirmed. We welcome the community to test on different hardware setups and share your results.

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

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 3. Improving VLM Inference Performance Using Inference Acceleration Frameworks

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This step mainly introduces how to use the MLX-VLM inference acceleration framework to improve the inference performance of PaddleOCR-VL.

### 3.1 Starting the VLM Inference Service

Install the MLX-VLM inference framework:

```shell
python -m pip install -U mlx-vlm
```

Start the MLX-VLM inference service:

```shell
mlx_vlm.server --port 8111
```

### 3.2 Client Usage Method

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

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 4. Service Deployment

Currently, only **manual deployment** is supported. Please refer to **Section 4.2 Method 2: Manual Deployment** in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.3 Client Invocation Methods

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 5. Model Fine-Tuning

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).
