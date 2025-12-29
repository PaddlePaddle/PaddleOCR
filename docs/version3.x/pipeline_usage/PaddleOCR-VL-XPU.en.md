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
    --priviledged \
    --shm-size 64g \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu \
    /bin/bash
# Call PaddleOCR CLI or Python API in the container
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu` in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu-offline`.

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

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md), making sure to specify `device='xpu'`.

## 3. Enhancing VLM Inference Performance Using Inference Acceleration Framework

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This step mainly introduces how to use the FastDeploy inference acceleration framework to enhance the inference performance of PaddleOCR-VL.

### 3.1 Starting the VLM Inference Service

PaddleOCR provides a Docker image for quickly starting the FastDeploy inference service. Use the following command to start the service (requires Docker version >= 19.03):

```shell
docker run \
    -it \
    --network host \
    --user root \
    --privileged \
    --shm-size 64g \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend fastdeploy
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu` in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu-offline`.

When launching the FastDeploy inference service, we provide a set of default parameter settings. If you need to adjust parameters such as GPU memory usage, you can configure additional parameters yourself. Please refer to [3.3.1 Server Parameter Adjustment](./PaddleOCR-VL.en.md#331-server-parameter-adjustment) to create a configuration file, then mount this file into the container and specify the configuration file using `backend_config` in the command to start the service, for example:

```shell
docker run \
    -it \
    --rm \
    --network host \
    --user root \
    --privileged \
    --shm-size 64G \
    -v fastdeploy_config.yml:/tmp/fastdeploy_config.yml \  
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/fastdeploy_config.yml
```

### 3.2 Client Usage Method

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 3.3 Performance Tuning

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 4. Service Deployment

>Please note that the PaddleOCR-VL service introduced in this section differs from the VLM inference service in the previous section: the latter is only responsible for one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

This step mainly introduces how to deploy PaddleOCR-VL as a service and call it using Docker Compose. The specific process is as follows:

1. Download the Compose file and the environment variable configuration file separately from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/accelerators/xpu/compose.yaml) and [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/accelerators/xpu/.env) to your local machine.

2. Execute the following command in the directory where `compose.yaml` and `.env` files are located to start the server, which listens on port **8080** by default:

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

Docker Compose starts two containers sequentially by reading configurations from the `.env` and `compose.yaml` files, running the underlying VLM inference service and the PaddleOCR-VL service (pipeline service) respectively.

The meanings of each environment variable contained in the `.env` file are as follows:

```
- `API_IMAGE_TAG_SUFFIX`: The tag suffix of the image used to launch the pipeline service.
- `VLM_BACKEND`: The VLM inference backend.
- `VLM_IMAGE_TAG_SUFFIX`: The tag suffix of the image used to launch the VLM inference service.
```

You can modify `compose.yaml` to meet custom requirements, for example:

<details>
<summary>1. Change the port of the PaddleOCR-VL service</summary>

Edit <code>paddleocr-vl-api.ports</code> in the <code>compose.yaml</code> file to change the port. For example, if you need to change the service port to 8111, make the following modifications:

```diff
  paddleocr-vl-api:
    ...
    ports:
-     - 8080:8080
+     - 8111:8080
    ...
```

</details>

<details>
<summary>2. Specify the XPU used by the PaddleOCR-VL service</summary>

Edit <code>environment</code> in the <code>compose.yaml</code> file to change the XPU used. For example, if you need to use card 1 for deployment, make the following modifications:

```diff
  paddleocr-vl-api:
    ...
    environment:
+     - XPU_VISIBLE_DEVICES: 1
    ...
  paddleocr-vlm-server:
    ...
    environment:
+     - XPU_VISIBLE_DEVICES: 1
    ...
```

</details>

<details>
<summary>3. Adjust VLM server configuration</summary>

If you want to adjust the VLM server configuration, refer to <a href="./PaddleOCR-VL.en.md#331-server-parameter-adjustment">3.3.1 Server Parameter Adjustment</a> to generate a configuration file.

After generating the configuration file, add the following <code>paddleocr-vlm-server.volumes</code> and <code>paddleocr-vlm-server.command</code> fields to your <code>compose.yaml</code>. Replace <code>/path/to/your_config.yaml</code> with your actual configuration file path.

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend fastdeploy --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. Adjust pipeline-related configurations (such as model path, batch size, deployment device, etc.)</summary>

Refer to the <a href="./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions">4.4 Pipeline Configuration Adjustment Instructions</a> section.

</details>

### 4.3 Client Invocation Methods

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 5. Model Fine-Tuning

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).
