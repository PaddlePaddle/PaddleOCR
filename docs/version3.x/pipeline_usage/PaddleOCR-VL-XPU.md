---
comments: true
---

# PaddleOCR-VL XPU 环境配置教程

本教程是 PaddleOCR-VL 昆仑芯 XPU 的环境配置教程，目的是完成相关的环境配置，环境配置完毕后请参考 [PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 使用 PaddleOCR-VL。

## 1. 环境准备

此步骤主要介绍如何搭建 PaddleOCR-VL 的运行环境，有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像。

- 方法二：手动安装 PaddlePaddle 和 PaddleOCR。

### 1.1 方法一：使用 Docker 镜像

我们推荐使用官方 Docker 镜像（要求 Docker 版本 >= 19.03）：

```shell
docker run \
    -it \
    --network host \
    --user root \
    --priviledged \
    --shm-size 64g \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu \
    /bin/bash
# 在容器中调用 PaddleOCR CLI 或 Python API
```

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu` 更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-xpu-offline`。

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
python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

> **请注意安装 3.2.1 及以上版本的飞桨框架，同时安装特殊版本的 safetensors。**

## 2. 快速开始

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md)相同章节，注意需要指定 `deivce="xpu"`。

## 3. 使用推理加速框架提升 VLM 推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。此步骤主要介绍如何使用 FastDeploy 推理加速框架来提升 PaddleOCR-VL 的推理性能。

### 3.1 启动 VLM 推理服务

PaddleOCR 提供了 Docker 镜像，用于快速启动 FastDeploy 推理服务。可使用以下命令启动服务（要求 Docker 版本 >= 19.03）：

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

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu` 更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-xpu-offline`。

启动 FastDeploy 推理服务时，我们提供了一套默认参数设置。如果您有调整显存占用等更多参数的需求，可以自行配置更多参数。请参考 [3.3.1 服务端参数调整](./PaddleOCR-VL.md#331-服务端参数调整) 创建配置文件，然后将该文件挂载到容器中，并在启动服务的命令中使用 `backend_config` 指定配置文件，例如：

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

### 3.2 客户端使用方法

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 3.3 性能调优

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 4. 服务化部署

>请注意，本节所介绍 PaddleOCR-VL 服务与上一节中的 VLM 推理服务有所区别：后者仅负责完整流程中的一个环节（即 VLM 推理），并作为前者的底层服务被调用。

此步骤主要介绍如何使用 Docker Compose 将 PaddleOCR-VL 部署为服务并调用，具体流程如下：


1. 分别从 [此处](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/accelerators/xpu/compose.yaml) 和 [此处](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/accelerators/xpu/.env) 获取 Compose 文件与环境变量配置文件并下载到本地。

2. 在 `compose.yaml` 和 `.env` 文件所在目录下执行以下命令启动服务器，默认监听 **8080** 端口：

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

此方式基于 FastDeploy 框架对 VLM 推理进行加速，更适合生产环境部署。

此外，使用此方式启动服务器后，除拉取镜像外，无需连接互联网。如需在离线环境中部署，可先在联网机器上拉取 Compose 文件中涉及的镜像，导出并传输至离线机器中导入，即可在离线环境下启动服务。

Docker Compose 通过读取 `.env` 和 `compose.yaml` 文件中配置，先后启动 2 个容器，分别运行底层 VLM 推理服务，以及 PaddleOCR-VL 服务（产线服务）。

`.env` 文件中包含的各环境变量含义如下：

- `API_IMAGE_TAG_SUFFIX`：启动产线服务使用的镜像的标签后缀。
- `VLM_BACKEND`：VLM 推理后端。
- `VLM_IMAGE_TAG_SUFFIX`：启动 VLM 推理服务使用的镜像的标签后缀。

您可以通过修改 `compose.yaml` 来满足自定义需求，例如：

<details>
<summary>1. 更改 PaddleOCR-VL 服务的端口</summary>

编辑 <code>compose.yaml</code> 文件中的 <code>paddleocr-vl-api.ports</code> 来更改端口。例如，如果您需要将服务端口更换为 8111，可以进行以下修改：

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
<summary>2. 指定 PaddleOCR-VL 服务所使用的 XPU</summary>

编辑 <code>compose.yaml</code> 文件中的 <code>environment</code> 来更改所使用的 XPU。例如，如果您需要使用卡 1 进行部署，可以进行以下修改：

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
<summary>3. 调整 VLM 服务端配置</summary>

若您想调整 VLM 服务端的配置，可以参考 <a href="./PaddleOCR-VL.md#331-服务端参数调整">3.3.1 服务端参数调整</a> 生成配置文件。

生成配置文件后，将以下的 <code>paddleocr-vlm-server.volumes</code> 和 <code>paddleocr-vlm-server.command</code> 字段增加到您的 <code>compose.yaml</code> 中。请将 <code>/path/to/your_config.yaml</code> 替换为您的实际配置文件路径。

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend fastdeploy --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. 调整产线相关配置（如模型路径、批处理大小、部署设备等）</summary>

参考 <a href="./PaddleOCR-VL.md#44-产线配置调整说明">4.4 产线配置调整说明</a> 小节。

</details>

### 4.3 客户端调用方式

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 4.4 产线配置调整说明

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。
