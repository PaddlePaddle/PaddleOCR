---
comments: true
---

# PaddleOCR-VL DCU Environment Configuration Tutorial

This tutorial is a guide for configuring the PaddleOCR-VL HYGON DCU environment. The purpose is to complete the relevant environment setup. After the environment configuration is complete, please refer to the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md) to use PaddleOCR-VL.

## 1. Environment Preparation

This step mainly introduces how to set up the runtime environment for PaddleOCR-VL. There are two methods available; choose either one:

- Method 1: Use the official Docker image.

- Method 2: Manually install PaddlePaddle and PaddleOCR.

### 1.1 Method 1: Using Docker Image

We recommend using the official Docker image (requires Docker version >= 19.03):

```shell
docker run -it \
  --rm \
  --user root \
  --privileged \
  --device /dev/kfd \
  --device /dev/dri \
  --device /dev/mkfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal/:/opt/hyhal/:ro \
  --shm-size=64G \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-dcu \
  /bin/bash
# Call PaddleOCR CLI or Python API in the container
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-dcu` (image size approximately 21 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-dcu-offline` (image size approximately 23 GB).

### 1.2 Method 2: Manually Install PaddlePaddle and PaddleOCR

If you cannot use Docker, you can also manually install PaddlePaddle and PaddleOCR. Python version 3.8–3.12 is required.

**We strongly recommend installing PaddleOCR-VL in a virtual environment to avoid dependency conflicts.** For example, use the Python venv standard library to create a virtual environment:

```shell
# Create a virtual environment
python -m venv .venv_paddleocr
# Activate the environment
source .venv_paddleocr/bin/activate
```

Execute the following commands to complete the installation:

```shell
python -m pip install paddlepaddle-dcu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

> **Please note to install PaddlePaddle version 3.2.1 or above, and install the special version of safetensors.**

## 2. Quick Start

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 3. Improving VLM Inference Performance Using Inference Acceleration Framework

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This step mainly introduces how to use the vLLM inference acceleration framework to improve the inference performance of PaddleOCR-VL.

### 3.1 Starting the VLM Inference Service

PaddleOCR provides a Docker image for quickly starting the vLLM inference service. Use the following command to start the service (requires Docker version >= 19.03):

```shell
docker run -it \
  --rm \
  --user root \
  --privileged \
  --device /dev/kfd \
  --device /dev/dri \
  --device /dev/mkfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal/:/opt/hyhal/:ro \
  --shm-size=64G \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-dcu \
  paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-dcu` (image size approximately 25 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-dcu-offline` (image size approximately 27 GB).

### 3.2 Client Usage Method

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 4. Service Deployment

>Please note that the PaddleOCR-VL service introduced in this section is different from the VLM inference service in the previous section: the latter is only responsible for one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

This step mainly introduces how to use Docker Compose to deploy PaddleOCR-VL as a service and call it. The specific process is as follows:

1. Copy the content from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/compose_dcu.yaml) and save it as a `compose.yaml` file.

2. Copy the following content and save it as a `.env` file:

    ```
    API_IMAGE_TAG_SUFFIX=latest-dcu-offline
    VLM_BACKEND=vllm
    VLM_IMAGE_TAG_SUFFIX=latest-dcu-offline
    ```3. Execute the following command in the directory where the `compose.yaml` and `.env` files are located to start the server, which listens on port **8080** by default:

    ```shell
    # Must be executed in the directory where compose.yaml and .env files are located
    docker compose up
    ```

    After startup, you will see output similar to the following:

    ```text
    paddleocr-vl-api             | INFO:     Started server process [1]
    paddleocr-vl-api             | INFO:     Waiting for application startup.
    paddleocr-vl-api             | INFO:     Application startup complete.
    paddleocr-vl-api             | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
    ```

This method accelerates VLM inference using the vLLM framework and is more suitable for production environment deployment.

Additionally, after starting the server in this manner, no internet connection is required except for image pulling. For deployment in an offline environment, you can first pull the images involved in the Compose file on a connected machine, export them, and transfer them to the offline machine for import to start the service in an offline environment.

To adjust pipeline configurations (such as model paths, batch sizes, deployment devices, etc.), refer to Section 4.4.

### 4.3 Client Invocation Method

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 5. Model Fine-Tuning

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).
