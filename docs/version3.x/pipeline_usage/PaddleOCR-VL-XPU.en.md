---
comments: true
---

# PaddleOCR-VL XPU Environment Configuration Tutorial

This tutorial is a guide for configuring the environment for PaddleOCR-VL KUNLUNXIN XPU. After completing the environment setup, please refer to the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md) to use PaddleOCR-VL.

## 1. Environment Preparation

This step mainly introduces how to set up the runtime environment for PaddleOCR-VL. There are two methods available; choose one as needed:

- Method 1: Use the official Docker image.

- Method 2: Manually install PaddlePaddle and PaddleOCR.

### 1.1 Method 1: Using Docker Image

We recommend using the official Docker image (requires Docker version >= 19.03):

```shell
docker run \
    -it \
    --network host \
    --user root \
    --shm-size 64G \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu \
    /bin/bash
# Call PaddleOCR CLI or Python API in the container
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu` (image size approximately 12 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu-offline` (image size approximately 14 GB).

### 1.2 Method 2: Manually Install PaddlePaddle and PaddleOCR

If you cannot use Docker, you can also manually install PaddlePaddle and PaddleOCR. The required Python version is 3.8–3.12.

**We strongly recommend installing PaddleOCR-VL in a virtual environment to avoid dependency conflicts.** For example, use the Python venv standard library to create a virtual environment:

```shell
# Create a virtual environment
python -m venv .venv_paddleocr
# Activate the environment
source .venv_paddleocr/bin/activate
```

Execute the following commands to complete the installation:

```shell
python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

> **Please ensure to install PaddlePaddle version 3.2.1 or above, along with the special version of safetensors.**

## 2. Quick Start

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 3. Enhancing VLM Inference Performance Using Inference Acceleration Framework

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This step mainly introduces how to use the FastDeploy inference acceleration framework to enhance the inference performance of PaddleOCR-VL.

### 3.1 Starting the VLM Inference Service

PaddleOCR provides a Docker image for quickly starting the FastDeploy inference service. Use the following command to start the service (requires Docker version >= 19.03):

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    --shm-size 64G \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend fastdeploy
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu` (image size approximately 47 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu-offline` (image size approximately 49 GB).

### 3.2 Client Usage Method

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 4. Service Deployment

>Please note that the PaddleOCR-VL service introduced in this section differs from the VLM inference service in the previous section: the latter is only responsible for one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

This step mainly introduces how to deploy PaddleOCR-VL as a service and call it using Docker Compose. The specific process is as follows:

1. Copy the content from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/compose_xpu.yaml) and save it as a `compose.yaml` file.

2. Copy the following content and save it as a `.env` file:

    ```
    API_IMAGE_TAG_SUFFIX=latest-xpu-offline
    VLM_BACKEND=fastdeploy
    VLM_IMAGE_TAG_SUFFIX=latest-xpu-offline
    ```

3. Execute the following command in the directory where `compose.yaml` and `.env` files are located to start the server, which listens on port **8080** by default:

    ```shell
    # Must be executed in the directory where compose.yaml and .env files are located
    docker compose up
    ```

    After starting, you will see output similar to the following:

    ```text
    paddleocr-vl-api             | INFO:     Started server process [1]
    paddleocr-vl-api             | INFO:     Waiting for application startup.
    paddleocr-vl-api             | INFO:     Application startup complete.
    paddleocr-vl-api             | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
    ```

This method accelerates VLM inference based on the FastDeploy framework and is more suitable for production environment deployment.

Additionally, after starting the server using this method, no internet connection is required except for pulling the image. If you need to deploy in an offline environment, you can first pull the images involved in the Compose file on a connected machine, export them, and transfer them to the offline machine for import to start the service in an offline environment.

If you need to adjust pipeline configurations (such as model path, batch size, deployment device, etc.), please refer to section 4.4.

### 4.3 Client Invocation Methods

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 5. Model Fine-Tuning

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).
