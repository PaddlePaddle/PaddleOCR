---
comments: true
---

# PaddleOCR-VL DCU 环境配置教程

本教程是 PaddleOCR-VL 海光 DCU 的环境配置教程，目的是完成相关的环境配置，环境配置完毕后请参考 [PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 使用 PaddleOCR-VL。

## 1. 环境准备

此步骤主要介绍如何搭建 PaddleOCR-VL 的运行环境，有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像。

- 方法二：手动安装 PaddlePaddle 和 PaddleOCR。

### 1.1 方法一：使用 Docker 镜像

我们推荐使用官方 Docker 镜像（要求 Docker 版本 >= 19.03）：

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
# 在容器中调用 PaddleOCR CLI 或 Python API
```

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-dcu`（镜像大小约为 21 GB）更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-dcu-offline`（镜像大小约为 23 GB）。

### 1.2 方法二：手动安装 PaddlePaddle 和 PaddleOCR

如果您无法使用 Docker，也可以手动安装 PaddlePaddle 和 PaddleOCR。要求 Python 版本为 3.8–3.12。

**我们强烈推荐您在虚拟环境中安装 PaddleOCR-VL，以避免发生依赖冲突。** 例如，使用 Python venv 标准库创建虚拟环境：

```shell
# 创建虚拟环境
python -m venv .venv_paddleocr
# 激活环境
source .venv_paddleocr/bin/activate
```

执行如下命令完成安装：

```shell
python -m pip install paddlepaddle-dcu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

> **请注意安装 3.2.1 及以上版本的飞桨框架，同时安装特殊版本的 safetensors。**

## 2. 快速开始

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md)相同章节。

## 3. 使用推理加速框架提升 VLM 推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。此步骤主要介绍如何使用 vLLM 推理加速框架来提升 PaddleOCR-VL 的推理性能。

### 3.1 启动 VLM 推理服务

PaddleOCR 提供了 Docker 镜像，用于快速启动 vLLM 推理服务。可使用以下命令启动服务（要求 Docker 版本 >= 19.03）：

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

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-dcu`（镜像大小约为 25 GB）更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-dcu-offline`（镜像大小约为 27 GB）。

### 3.2 客户端使用方法

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 4. 服务化部署

>请注意，本节所介绍 PaddleOCR-VL 服务与上一节中的 VLM 推理服务有所区别：后者仅负责完整流程中的一个环节（即 VLM 推理），并作为前者的底层服务被调用。

此步骤主要介绍如何使用 Docker Compose 将 PaddleOCR-VL 部署为服务并调用，具体流程如下：

1. 从 [此处](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/compose_dcu.yaml) 复制内容保存为 `compose.yaml` 文件。

2. 复制以下内容并保存为 `.env` 文件：

    ```
    API_IMAGE_TAG_SUFFIX=latest-dcu-offline
    VLM_BACKEND=vllm
    VLM_IMAGE_TAG_SUFFIX=latest-dcu-offline
    ```

3. 在 `compose.yaml` 和 `.env` 文件所在目录下执行以下命令启动服务器，默认监听 **8080** 端口：

    ```shell
    # 必须在 compose.yaml 和 .env 文件所在的目录中执行
    docker compose up
    ```

    启动后将看到类似如下输出：

    ```text
    paddleocr-vl-api             | INFO:     Started server process [1]
    paddleocr-vl-api             | INFO:     Waiting for application startup.
    paddleocr-vl-api             | INFO:     Application startup complete.
    paddleocr-vl-api             | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
    ```

此方式基于 vLLM 框架对 VLM 推理进行加速，更适合生产环境部署。

此外，使用此方式启动服务器后，除拉取镜像外，无需连接互联网。如需在离线环境中部署，可先在联网机器上拉取 Compose 文件中涉及的镜像，导出并传输至离线机器中导入，即可在离线环境下启动服务。

如需调整产线相关配置（如模型路径、批处理大小、部署设备等），可参考 4.4 小节。

### 4.3 客户端调用方式

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 4.4 产线配置调整说明

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。
