---
comments: true
---

# PaddleOCR-VL NVIDIA SM120 GPU Usage Tutorial

> INFO:
> Unless otherwise specified, the term "PaddleOCR-VL" in this tutorial refers to the PaddleOCR-VL model series (e.g., PaddleOCR-VL-1.6). References specific to the PaddleOCR-VL v1 version will be explicitly noted.

Use this guide to set up local PaddlePaddle inference and the CUDA 12.9 API image on GPUs with compute capability 12.0 (SM120). These include GeForce RTX 50-series and RTX PRO Blackwell models. Other Blackwell GPUs have different compute capabilities, including B200 (10.0), B300 (10.3), and GB10 (12.1). See the [NVIDIA GPU compute capability table](https://developer.nvidia.com/cuda/gpus).

The earlier PaddleOCR-VL accuracy and speed validation used an RTX 5070. The vLLM service uses the shared NVIDIA VLM image.

Before starting the tutorial, **please ensure that your NVIDIA driver supports CUDA 12.9 or higher**.

## Workflow Guide for This Hardware

Use this guide for the workflows below.

| Goal | Support on this hardware | Read this section |
| --- | --- | --- |
| Local direct inference | Supported | Read Section 1. Local Runtime Environment Preparation and Section 2. Quick Start. |
| Client + VLM inference service | Supported | Complete local direct inference first, then read Section 3. Using VLM Inference Services. |
| Full API service | Supported with Docker Compose or manual deployment | Use Section 4.1 for Docker Compose, or Section 4.2 for manual deployment (complete Section 1. Local Runtime Environment Preparation first), then continue with the Section 4.3 client invocation section and the Section 4.4 pipeline configuration section. |
| Model fine-tuning | Supported | Read Section 5. Model Fine-Tuning. |

If you only need to confirm which inference methods are available on this hardware, refer to the [PaddleOCR-VL Inference Method and Hardware Support Matrix](./PaddleOCR-VL.en.md#inference-device-support-for-paddleocr-vl) in the main guide.

## 1. Local Runtime Environment Preparation

**Local Runtime Environment Setup Methods Supported on This Hardware**

| Local runtime environment setup method | Status | Notes |
| --- | --- | --- |
| Official Docker image | Supported with steps in this guide | Continue with Section 1.1. |
| Manually install the inference engine and PaddleOCR | Supported with steps in this guide | Continue with Section 1.2. |

This section introduces how to set up the PaddleOCR-VL local runtime environment using one of the following two methods:

- Method 1: Use the official Docker image.

- Method 2: Manually install the inference engine and PaddleOCR.

**We strongly recommend using the Docker image to minimize potential environment-related issues.**

### 1.1 Method 1: Using Docker Image

We recommend using the official Docker image (requires Docker version >= 19.03, GPU-equipped machine with NVIDIA driver supporting CUDA 12.9 or higher):

```shell
docker run \
    -it \
    --gpus all \
    --network host \
    --user root \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 \
    /bin/bash
# Call PaddleOCR CLI or Python API in the container
```

If you wish to use PaddleOCR-VL in an offline environment, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120` (image size approximately 10 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline` (image size approximately 12 GB).

> TIP:
> Images with the `latest-xxx` tag correspond to the latest version.
> If the corresponding `latest` image already exists locally and you want the newest features or fixes, we recommend running `docker pull` again before using it.
> If you want to use an image corresponding to a specific PaddleOCR version, you can replace `latest` in the tag with the desired version number: `paddleocr<major>.<minor>`.
> For example:
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:paddleocr3.3-nvidia-gpu-sm120-offline`


### 1.2 Method 2: Manually Install the Inference Engine and PaddleOCR

If Docker is not an option, you can manually install the inference engine and PaddleOCR. This guide documents Python 3.9–3.13 as the verified range.

This guide provides PaddlePaddle installation steps. To use Transformers or other inference engines, see [Section 1.2 of the main tutorial](./PaddleOCR-VL.en.md#manual-install-inference-engine-and-paddleocr).

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
```

> **Please ensure that PaddlePaddle framework version 3.2.1 or higher is installed.**

## 2. Quick Start

Please refer to [PaddleOCR-VL Usage Tutorial - 2. Quick Start](./PaddleOCR-VL.en.md#2-quick-start).

## 3. Using VLM Inference Services

This section shows how to start the vLLM inference service on an SM120 GPU.

### 3.1 Starting the VLM Inference Service

> IMPORTANT:
> The service started according to this section is responsible only for the VLM inference stage in the PaddleOCR-VL workflow. It does not provide a complete end-to-end document parsing API. We strongly recommend that you do not call this service directly via HTTP requests or OpenAI clients to process document images. If you need to deploy a service with the full PaddleOCR-VL capabilities, refer to the service deployment section later in this document.

**Launch Methods Supported on This Hardware**

| Launch method | Status | Notes |
| --- | --- | --- |
| Official Docker image | Supported with steps in this guide | Continue with Section 3.1.1. |
| Install dependencies with the PaddleOCR CLI and launch the service | Supported with steps in this guide | Continue with Section 3.1.2. |
| Launch the service directly with the acceleration framework | Not verified | See the [main guide](./PaddleOCR-VL.en.md#3-vlm) for native vLLM startup. |

There are two methods to start the VLM inference service; choose one:

- Method 1: Start the service using the official Docker image.

- Method 2: Manually install dependencies and start the service via PaddleOCR CLI.

**We strongly recommend using the Docker image to minimize potential environment-related issues.**

#### 3.1.1 Method 1: Using Docker Image

PaddleOCR provides a Docker image for quickly starting the vLLM inference service. Use the following command to start the service (requires Docker version >= 19.03, GPU-equipped machine with NVIDIA driver supporting CUDA 12.9 or higher):

```shell
docker run \
    -it \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server --model_name PaddleOCR-VL-1.8-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

If you wish to start the service in an offline environment, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu` in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline`.

When launching the vLLM inference service, we provide a set of default parameter settings. If you need to adjust parameters such as GPU memory usage, you can configure additional parameters yourself. Please refer to [3.3.1 Server-side Parameter Adjustment](./PaddleOCR-VL.en.md#331-server-side-parameter-adjustment) to create a configuration file, then mount the file into the container and specify the configuration file using `backend_config` in the command to start the service, for example:

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    -v ./vllm_config.yml:/tmp/vllm_config.yml \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server --model_name PaddleOCR-VL-1.8-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/vllm_config.yml
```

> TIP:
> Images with the `latest-xxx` tag correspond to the latest version.
> If the corresponding `latest` image already exists locally and you want the newest features or fixes, we recommend running `docker pull` again before using it.
> If you want to use an image corresponding to a specific PaddleOCR version, you can replace `latest` in the tag with the desired version number: `paddleocr<major>.<minor>`.
> For example:
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:paddleocr3.8-nvidia-gpu-offline`


#### 3.1.2 Method 2: Installation and Usage via PaddleOCR CLI

Since inference acceleration frameworks may conflict with packages already installed in the current environment, it is recommended to install them in a virtual environment:

```shell
# If a virtual environment is currently activated, deactivate it first using `deactivate`
# Create a virtual environment
python -m venv .venv_vlm
# Activate the environment
source .venv_vlm/bin/activate
```

The shared vLLM setup below uses Python 3.10–3.13 and CUDA 12.9.

Install PaddleOCR and the dependencies of inference acceleration services, using vLLM as an example:

```shell
# Install PaddleOCR
python -m pip install paddleocr
# Install inference acceleration service dependencies
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129 paddleocr install_genai_server_deps vllm
```

Usage of the `paddleocr install_genai_server_deps` command:

```shell
paddleocr install_genai_server_deps <inference acceleration framework name>
```

This guide uses the `vllm` backend.

> WARNING:
> If you use Transformers for layout analysis, use separate Python environments for layout analysis and the vLLM service.

After installation, you can start the service using the `paddleocr genai_server` command:

```shell
paddleocr genai_server --model_name PaddleOCR-VL-1.8-0.9B --backend vllm --port 8118
```

The parameters supported by this command are as follows:

| Parameter         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--model_name`   | Name of the model                                                            |
| `--model_dir`    | Directory containing the model                                               |
| `--host`         | Server hostname                                                             |
| `--port`         | Server port number                                                           |
| `--backend`      | Use the `vllm` backend |
| `--backend_config`| YAML file specifying backend configuration                                   |

### 3.2 Client Usage

For client-side invocation methods, please refer to [PaddleOCR-VL Usage Tutorial - 3.2 Client Usage Methods](./PaddleOCR-VL.en.md#32-client-usage-methods). If you run the client on this hardware, make sure to specify `device="gpu"`.

### 3.3 Performance Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 3.3 Performance Tuning](./PaddleOCR-VL.en.md#33-performance-tuning).

## 4. Service Deployment

**Deployment Methods Supported on This Hardware**

| Deployment method | Status | Notes |
| --- | --- | --- |
| Docker Compose deployment | Supported with steps in this guide | Continue with Section 4.1. |
| Manual deployment | Supported | Complete Section 1. Local Runtime Environment Preparation first, then continue with Section 4.2. |

This section mainly introduces how to deploy PaddleOCR-VL as a service and invoke it. There are two methods available; choose one:

- Method 1: Deploy using Docker Compose.

- Method 2: Manually install dependencies for deployment.

> IMPORTANT:
> The PaddleOCR-VL service introduced in this section differs from the VLM inference service in the previous section: the latter is responsible for only one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

<a id="sm120-compose"></a>
### 4.1 Method 1: Deploy Using Docker Compose

1. Download the Compose file and the environment variable configuration file separately from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/compose.yaml) and [here](https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/.env) to your local machine.

2. Execute the following command in the directory containing the `compose.yaml` and `.env` files to start the server, which will listen on port **8080** by default:

    ```shell
    # Must be executed in the directory containing compose.yaml and .env files
    docker compose up
    ```

    > TIP:
    > In the supplied `.env`, `API_IMAGE_TAG_SUFFIX` selects the SM120 API image (`latest-nvidia-gpu-sm120-offline`). `VLM_IMAGE_TAG_SUFFIX` selects the shared VLM image (`latest-nvidia-gpu-offline`).
    > For version 3.8, use `paddleocr3.8-nvidia-gpu-sm120-offline` for the API and `paddleocr3.8-nvidia-gpu-offline` for the VLM.

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

If you want to adjust the VLM server configuration, refer to <a href="./PaddleOCR-VL.en.md#331-server-side-parameter-adjustment">3.3.1 Server Parameter Adjustment</a> to generate a configuration file.

After generating the configuration file, add the following <code>paddleocr-vlm-server.volumes</code> and <code>paddleocr-vlm-server.command</code> fields to your <code>compose.yaml</code>. Replace <code>/path/to/your_config.yaml</code> with your actual configuration file path.

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-1.8-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. Adjust pipeline-related configurations (such as model path, batch size, deployment device, etc.)</summary>

Refer to the <a href="./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions">4.4 Pipeline Configuration Adjustment Instructions</a> section.

</details>

### 4.2 Method 2: Manually Deployment

Please complete Section 1. Local Runtime Environment Preparation first, then refer to [PaddleOCR-VL Usage Tutorial - 4.2 Method 2: Manual Deployment](./PaddleOCR-VL.en.md#42-method-2-manual-deployment).

### 4.3 Client Invocation Methods

Please refer to [PaddleOCR-VL Usage Tutorial - 4.3 Client Invocation Methods](./PaddleOCR-VL.en.md#43-client-side-invocation).

### 4.4 Pipeline Configuration Adjustment Instructions

Please refer to [PaddleOCR-VL Usage Tutorial - 4.4 Pipeline Configuration Adjustment Instructions](./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions).

## 5. Model Fine-Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 5. Model Fine-Tuning](./PaddleOCR-VL.en.md#5-model-fine-tuning).
