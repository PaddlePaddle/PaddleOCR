---
comments: true
---

# PaddleOCR-VL Hygon DCU Usage Tutorial

> INFO:
> Unless otherwise specified, the term "PaddleOCR-VL" in this tutorial refers to the PaddleOCR-VL model series (e.g., PaddleOCR-VL-1.5). References specific to the PaddleOCR-VL v1 version will be explicitly noted.

This tutorial is a guide for using PaddleOCR-VL on Hygon DCU, covering the complete workflow from environment preparation to service deployment.

PaddleOCR-VL has been verified for accuracy and speed on the Hygon K100AI. However, due to hardware diversity, compatibility with other Hygon DCUs has not yet been confirmed. We welcome the community to test on different hardware setups and share your results.

## Workflow Guide for This Hardware

Use this guide for the workflows below.

| Goal | Support on this hardware | Read this section |
| --- | --- | --- |
| Local direct inference | Supported | Read Section 1. Environment Preparation and Section 2. Quick Start. |
| Client + VLM inference service | Supported | Complete local direct inference first, then read Section 3. Improving Inference Performance with VLM Inference Services. |
| Full API service | Supported with Docker Compose deployment | Read Section 4.1 first, then continue with the Section 4.2 client invocation section and the Section 4.3 pipeline configuration section. |
| Model fine-tuning | Supported | Read Section 5. Model Fine-Tuning. |

If you only need to confirm which inference methods are available on this hardware, refer to the [PaddleOCR-VL Inference Method and Hardware Support Matrix](./PaddleOCR-VL.en.md#inference-device-support-for-paddleocr-vl) in the main guide.

## 1. Environment Preparation

**Environment Setup Methods Supported on This Hardware**

| Environment setup method | Status | Notes |
| --- | --- | --- |
| Official Docker image | Supported with steps in this guide | Continue with Section 1.1. |
| Manually install PaddlePaddle and PaddleOCR | Supported with steps in this guide | Continue with Section 1.2. |

This step mainly introduces how to set up the runtime environment for PaddleOCR-VL. There are two methods available; choose either one:

- Method 1: Use the official Docker image.

- Method 2: Manually install PaddlePaddle and PaddleOCR.

**We strongly recommend using the Docker image to minimize potential environment-related issues.**

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
  --shm-size 64g \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-hygon-dcu \
  /bin/bash
# Call PaddleOCR CLI or Python API in the container
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-hygon-dcu` (image size approximately 22 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-hygon-dcu-offline` (image size approximately 24 GB).

> TIP:
> Images with the `latest-xxx` tag correspond to the latest version.
> If the corresponding `latest` image already exists locally and you want the newest features or fixes, we recommend running `docker pull` again before using it.
> If you want to use an image corresponding to a specific PaddleOCR version, you can replace `latest` in the tag with the desired version number: `paddleocr<major>.<minor>`.
> For example:
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:paddleocr3.3-hygon-dcu-offline`


### 1.2 Method 2: Manually Install PaddlePaddle and PaddleOCR

If you cannot use Docker, you can also manually install PaddlePaddle and PaddleOCR. This guide documents Python 3.9–3.13 as the verified range.

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
```

> **Please note to install PaddlePaddle version 3.2.1 or above.**

## 2. Quick Start

Please refer to [PaddleOCR-VL Usage Tutorial - 2. Quick Start](./PaddleOCR-VL.en.md#2-quick-start), making sure to specify `device="dcu"`.

## 3. Improving Inference Performance with VLM Inference Services

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This section introduces how to improve PaddleOCR-VL inference performance through a VLM inference service. In this hardware-specific guide, the examples use vLLM as the backend for the VLM inference service.

### 3.1 Starting the VLM Inference Service

> IMPORTANT:
> The service started according to this section is responsible only for the VLM inference stage in the PaddleOCR-VL workflow. It does not provide a complete end-to-end document parsing API. We strongly recommend that you do not call this service directly via HTTP requests or OpenAI clients to process document images. If you need to deploy a service with the full PaddleOCR-VL capabilities, refer to the service deployment section later in this document.

**Launch Methods Supported on This Hardware**

| Launch method | Status | Notes |
| --- | --- | --- |
| Official Docker image | Supported with steps in this guide | This section provides the vLLM service launch steps. |
| Install dependencies with the PaddleOCR CLI and launch the service | Not currently supported | This hardware does not currently support this path. |
| Launch the service directly with the acceleration framework | Not currently supported | This hardware does not currently support this path. |

PaddleOCR provides a Docker image for quickly starting the vLLM inference service. Use the following command to start the service (requires Docker version >= 19.03):

```shell
docker run -it \
  --user root \
  --privileged \
  --device /dev/kfd \
  --device /dev/dri \
  --device /dev/mkfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal/:/opt/hyhal/:ro \
  --shm-size 64g \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-hygon-dcu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

If you wish to start the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-hygon-dcu` (image size approximately 25 GB) in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-hygon-dcu-offline` (image size approximately 27 GB).

When launching the vLLM inference service, we provide a set of default parameter settings. If you need to adjust parameters such as GPU memory usage, you can configure additional parameters yourself. Please refer to [3.3.1 Server-side Parameter Adjustment](./PaddleOCR-VL.en.md#331-server-side-parameter-adjustment) to create a configuration file, then mount the file into the container and specify the configuration file using `backend_config` in the command to start the service, for example:

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
  -v vllm_config.yml:/tmp/vllm_config.yml \
  --shm-size 64g \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-dcu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/vllm_config.yml
```

> TIP:
> Images with the `latest-xxx` tag correspond to the latest version.
> If the corresponding `latest` image already exists locally and you want the newest features or fixes, we recommend running `docker pull` again before using it.
> If you want to use an image corresponding to a specific PaddleOCR version, you can replace `latest` in the tag with the desired version number: `paddleocr<major>.<minor>`.
> For example:
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:paddleocr3.3-hygon-dcu-offline`


### 3.2 Client Usage Method

For client-side invocation methods, please refer to [PaddleOCR-VL Usage Tutorial - 3.2 Client Usage Methods](./PaddleOCR-VL.en.md#32-client-usage-methods). If you run the client on this hardware, make sure to specify `device="dcu"`.

### 3.3 Performance Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 3.3 Performance Tuning](./PaddleOCR-VL.en.md#33-performance-tuning).

## 4. Service Deployment

**Deployment Methods Supported on This Hardware**

| Deployment method | Status | Notes |
| --- | --- | --- |
| Docker Compose deployment | Supported with steps in this guide | Continue with Section 4.1. |
| Manual deployment | Not currently supported | This hardware does not currently support this path. |

> IMPORTANT:
> The PaddleOCR-VL service introduced in this section differs from the VLM inference service in the previous section: the latter is responsible for only one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

### 4.1 Deploy Using Docker Compose

This step mainly introduces how to use Docker Compose to deploy PaddleOCR-VL as a service and call it. The specific process is as follows:

1. Download the Compose file and the environment variable configuration file separately from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/accelerators/hygon-dcu/compose.yaml) and [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/accelerators/hygon-dcu/.env) to your local machine.
    
2. Execute the following command in the directory where the `compose.yaml` and `.env` files are located to start the server, which listens on port **8080** by default:

    ```shell
    # Must be executed in the directory where compose.yaml and .env files are located
    docker compose up
    ```

    > TIP:
    > The image tags used by `compose.yaml` are usually controlled by `API_IMAGE_TAG_SUFFIX` and `VLM_IMAGE_TAG_SUFFIX` in `.env`, and default to tags such as `latest-hygon-dcu-offline`. To make sure you pull the newest `latest` images, run `docker compose pull` in the current directory before `docker compose up`.
    > To use an image corresponding to a specific PaddleOCR version, replace `latest` in these variables with `paddleocr<major>.<minor>`, for example `paddleocr3.3-hygon-dcu-offline`.

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
<summary>2. Specify the DCU used by the PaddleOCR-VL service</summary>

Edit <code>environment</code> in the <code>compose.yaml</code> file to change the DCU used. For example, if you need to use card 1 for deployment, make the following modifications:

```diff
  paddleocr-vl-api:
    ...
    environment:
+     - HIP_VISIBLE_DEVICES: 1
    ...
  paddleocr-vlm-server:
    ...
    environment:
+     - HIP_VISIBLE_DEVICES: 1
    ...
```


</details>

<details>
<summary>3. Adjust VLM server-side configuration</summary>

If you want to adjust the VLM server configuration, refer to <a href="./PaddleOCR-VL.en.md#331-server-side-parameter-adjustment">3.3.1 Server-side Parameter Adjustment</a> to generate a configuration file.

After generating the configuration file, add the following <code>paddleocr-vlm-server.volumes</code> and <code>paddleocr-vlm-server.command</code> fields to your <code>compose.yaml</code>. Replace <code>/path/to/your_config.yaml</code> with your actual configuration file path.

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. Adjust pipeline-related configurations (such as model path, batch size, deployment device, etc.)</summary>

Refer to the <a href="./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions">4.4 Pipeline Configuration Adjustment Instructions</a> section.

</details>

### 4.2 Client Invocation Method

Please refer to [PaddleOCR-VL Usage Tutorial - 4.3 Client-Side Invocation](./PaddleOCR-VL.en.md#43-client-side-invocation).

### 4.3 Pipeline Configuration Adjustment Instructions

Please refer to [PaddleOCR-VL Usage Tutorial - 4.4 Pipeline Configuration Adjustment Instructions](./PaddleOCR-VL.en.md#44-pipeline-configuration-adjustment-instructions).

## 5. Model Fine-Tuning

Please refer to [PaddleOCR-VL Usage Tutorial - 5. Model Fine-Tuning](./PaddleOCR-VL.en.md#5-model-fine-tuning).
