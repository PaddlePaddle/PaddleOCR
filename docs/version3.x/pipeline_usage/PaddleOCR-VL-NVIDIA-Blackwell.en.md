---
comments: true
---

# PaddleOCR-VL NVIDIA Blackwell-Architecture GPUs Environment Configuration Tutorial

This tutorial provides guidance on configuring the environment for NVIDIA Blackwell-architecture GPUs. After completing the environment setup, please refer to the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md) to use PaddleOCR-VL.

NVIDIA Blackwell-architecture GPUs include, but are not limited to:

- RTX 5090
- RTX 5080
- RTX 5070、RTX 5070 Ti
- RTX 5060、RTX 5060 Ti
- RTX 5050

Before starting the tutorial, **please ensure that your NVIDIA driver supports CUDA 12.9 or higher**.

## 1. Environment Preparation

This section introduces how to set up the PaddleOCR-VL runtime environment using one of the following two methods:

- Method 1: Use the official Docker image.

- Method 2: Manually install PaddlePaddle and PaddleOCR.

### 1.1 Method 1: Using Docker Image

We recommend using the official Docker image (requires Docker version >= 19.03, GPU-equipped machine with NVIDIA driver supporting CUDA 12.9 or higher):

```shell
docker run \
    -it \
    --gpus all \
    --network host \
    --user root \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-gpu-sm120 \
    /bin/bash
# Call PaddleOCR CLI or Python API in the container
```

If you wish to use PaddleOCR-VL in an offline environment, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-gpu-sm120` (image size ~10 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-gpu-sm120-offline` (image size ~12 GB).

### 1.2 Method 2: Manually Install PaddlePaddle and PaddleOCR

If Docker is not an option, you can manually install PaddlePaddle and PaddleOCR. Python version 3.8–3.12 is required.

**We strongly recommend installing PaddleOCR-VL in a virtual environment to avoid dependency conflicts.** For example, create a virtual environment using Python's standard venv library:

```shell
# Create a virtual environment
python -m venv .venv_paddleocr
# Activate the environment
source .venv_paddleocr/bin/activate
```

Run the following commands to complete the installation:

```shell
# Note that PaddlePaddle for cu129 is being installed here
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install -U "paddleocr[doc-parser]"
# For Linux systems, run:
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
# For Windows systems, run:
python -m pip install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl
```

> **Please ensure that PaddlePaddle framework version 3.2.1 or higher is installed, along with the special version of safetensors.**

## 2. Quick Start

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 3. Improving VLM Inference Performance Using Inference Acceleration Frameworks

The inference performance under default configurations may not be fully optimized and may not meet actual production requirements. This section introduces how to use the vLLM and SGLang inference acceleration frameworks to enhance PaddleOCR-VL's inference performance.

### 3.1 Starting the VLM Inference Service

There are two methods to start the VLM inference service; choose one:

- Method 1: Start the service using the official Docker image.

- Method 2: Manually install dependencies and start the service via PaddleOCR CLI.

#### 3.1.1 Method 1: Using Docker Image

PaddleOCR provides a Docker image for quickly starting the vLLM inference service. Use the following command to start the service (requires Docker version >= 19.03, GPU-equipped machine with NVIDIA driver supporting CUDA 12.9 or higher):

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120 \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

If you wish to start the service in an offline environment, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120` (image size ~12 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120-offline` (image size ~14 GB).

When launching the vLLM inference service, we provide a set of default parameter settings. If you need to adjust parameters such as GPU memory usage, you can configure additional parameters yourself. Please refer to [3.3.1 Server-side Parameter Adjustment](./PaddleOCR-VL.md#331-server-side-parameter-adjustment) to create a configuration file, then mount the file into the container and specify the configuration file using `backend_config` in the command to start the service, for example:

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    -v vllm_config.yml:/tmp/vllm_config.yml \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120 \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/vllm_config.yml
```

#### 3.1.2 Method 2: Installation and Usage via PaddleOCR CLI

Since inference acceleration frameworks may have dependency conflicts with the PaddlePaddle framework, installation in a virtual environment is recommended. Taking vLLM as an example:

```shell
# If there is an active virtual environment, deactivate it first using `deactivate`
# Create a virtual environment
python -m venv .venv_vlm
# Activate the environment
source .venv_vlm/bin/activate
# Install PaddleOCR
python -m pip install "paddleocr[doc-parser]"# Install dependencies for inference acceleration services
paddleocr install_genai_server_deps vllm
python -m pip install flash-attn==2.8.3
```

> The `paddleocr install_genai_server_deps` command may require CUDA compilation tools such as nvcc during execution. If these tools are not available in your environment or the installation takes too long, you can obtain a pre-compiled version of FlashAttention from [this repository](https://github.com/mjun0812/flash-attention-prebuild-wheels). For example, run `python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl`.

Usage of the `paddleocr install_genai_server_deps` command:

```shell
paddleocr install_genai_server_deps <inference acceleration framework name>
```

Currently supported framework names are `vllm` and `sglang`, corresponding to vLLM and SGLang, respectively.

After installation, you can start the service using the `paddleocr genai_server` command:

```shell
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

The parameters supported by this command are as follows:

| Parameter         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--model_name`   | Name of the model                                                            |
| `--model_dir`    | Directory containing the model                                               |
| `--host`         | Server hostname                                                             |
| `--port`         | Server port number                                                           |
| `--backend`      | Backend name, i.e., the name of the inference acceleration framework being used; options are `vllm` or `sglang` |
| `--backend_config`| YAML file specifying backend configuration                                   |

### 3.2 Client Usage

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 4. Service Deployment

This section mainly introduces how to deploy PaddleOCR-VL as a service and invoke it. There are two methods available; choose one:

- Method 1: Deploy using Docker Compose.

- Method 2: Manually install dependencies for deployment.

Please note that the PaddleOCR-VL service introduced in this section differs from the VLM inference service in the previous section: the latter is responsible for only one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

### 4.1 Method 1: Deploy Using Docker Compose

1. Copy the content from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/compose.yaml) and save it as a `compose.yaml` file.

2. Copy the following content and save it as a `.env` file:

    ```
    API_IMAGE_TAG_SUFFIX=latest-gpu-sm120-offline
    VLM_BACKEND=vllm
    VLM_IMAGE_TAG_SUFFIX=latest-gpu-sm120-offline
    ```

3. Execute the following command in the directory containing the `compose.yaml` and `.env` files to start the server, which will listen on port **8080** by default:

    ```shell
    # Must be executed in the directory containing compose.yaml and .env files
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
<summary>2. Specify the GPU used by the PaddleOCR-VL service</summary>

Edit <code>environment</code> in the <code>compose.yaml</code> file to change the GPU used. For example, if you need to use card 1 for deployment, make the following modifications:

```diff
  paddleocr-vl-api:
    ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
-             device_ids: ["0"]
+             device_ids: ["1"]
              capabilities: [gpu]
    ...
  paddleocr-vlm-server:
    ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
-             device_ids: ["0"]
+             device_ids: ["1"]
              capabilities: [gpu]
    ...
```

</details>

<details>
<summary>3. Adjust VLM server-side configuration</summary>

If you want to adjust the VLM server configuration, refer to <a href="./PaddleOCR-VL.en.md#331-server-parameter-adjustment">3.3.1 Server Parameter Adjustment</a> to generate a configuration file.

After generating the configuration file, add the following <code>paddleocr-vlm-server.volumes</code> and <code>paddleocr-vlm-server.command</code> fields to your <code>compose.yaml</code>. Replace <code>/path/to/your_config.yaml</code> with your actual configuration file path.

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. Adjust pipeline-related configurations (such as model path, batch size, deployment device, etc.)</summary>

Refer to the <a href="./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions">4.4 Pipeline Configuration Adjustment Instructions</a> section.

</details>

### 4.2 Method 2: Manually Deployment

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.3 Client Invocation Methods

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).

## 5. Model Fine-Tuning

Please refer to the corresponding section in the [PaddleOCR-VL Usage Tutorial](./PaddleOCR-VL.en.md).
