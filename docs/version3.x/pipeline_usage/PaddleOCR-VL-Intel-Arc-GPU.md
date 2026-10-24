---
comments: true
---

# PaddleOCR-VL Intel Arc GPU 使用教程

> INFO:
> 除非另有说明，本教程中提到的 “PaddleOCR-VL” 均指 PaddleOCR-VL 系列模型（如 PaddleOCR-VL-1.6 等）；若特指 PaddleOCR-VL v1 版本，将另行明确标注。

本教程是 PaddleOCR-VL 在 Intel Arc GPU 上的使用指南，涵盖了从本地运行环境准备到服务化部署的完整流程。

目前 PaddleOCR-VL 已在 Intel Arc B60 Pro 上完成精度、速度验证；鉴于硬件环境的多样性，其他 Intel Arc GPU 的兼容性尚未验证。我们诚挚欢迎社区用户在不同硬件上进行测试并反馈您的运行结果。

## 本硬件支持的使用目标

请在本硬件教程中按下表继续阅读。

| 目标 | 本硬件上的支持情况 | 从哪里开始阅读 |
| --- | --- | --- |
| 本地直接推理 | 当前不支持“本地直接推理”路径 | 请改走“客户端 + VLM 推理服务”路径：从第 1 节“本地运行环境准备”开始，然后阅读第 3 节。 |
| 客户端 + VLM 推理服务 | 支持 | 从第 1 节“本地运行环境准备”开始，然后阅读第 3 节“使用 VLM 推理服务”。 |
| 完整 API 服务 | 支持 Docker Compose 部署 | 先阅读第 4.1 节，再继续阅读第 4.2 节客户端调用部分和第 4.3 节产线配置调整部分。 |
| 模型微调 | 支持 | 阅读第 5 节“模型微调”。 |

如果你只是想先确认本硬件支持哪些推理方式，请参考主教程中的 [PaddleOCR-VL 推理方式与硬件支持矩阵](./PaddleOCR-VL.md#paddleocr-vl-对推理设备的支持情况)。

## 1. 本地运行环境准备

**当前硬件支持的本地运行环境准备方式**

| 本地运行环境准备方式 | 状态 | 说明 |
| --- | --- | --- |
| 官方 Docker 镜像 | 支持并提供步骤 | 请继续阅读本节的 1.1。 |
| 手动安装推理引擎和 PaddleOCR | 支持并提供步骤 | 请继续阅读本节的 1.2。 |

此步骤主要介绍如何搭建 PaddleOCR-VL 的本地运行环境，有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像。

- 方法二：手动安装推理引擎和 PaddleOCR。

**我们强烈推荐采用 Docker 镜像的方式，以最大程度减少可能出现的环境问题。**

### 1.1 方法一：使用 Docker 镜像

我们推荐使用官方 Docker 镜像（要求 Docker 版本 >= 19.03）：

```shell
docker run -it \
  --user root \
  --device /dev:/dev \
  --shm-size 64g \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-intel-gpu \
  /bin/bash
# 在容器中调用 PaddleOCR CLI 或 Python API
```

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-intel-gpu`（镜像的大小约为 31 GB）更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-intel-gpu-offline`（镜像的大小约为 33 GB）。

> TIP:
> 标签后缀为 `latest-xxx` 的镜像对应最新版本。
> 如果本地已经存在对应的 `latest` 镜像，但希望使用最新功能或修复，建议在继续使用前重新执行一次 `docker pull` 更新镜像。
> 如果希望使用特定版本 PaddleOCR 对应的镜像，可以将标签中的 `latest` 替换为对应版本号：`paddleocr<major>.<minor>`。
> 例如：
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:paddleocr3.4-intel-gpu-offline`

### 1.2 方法二：手动安装推理引擎和 PaddleOCR

如果您无法使用 Docker，也可以手动安装推理引擎和 PaddleOCR。本文档验证过的 Python 版本范围为 3.9–3.13。

当前硬件本地推理仅提供 PaddlePaddle 安装步骤，其他推理引擎尚在适配验证中。

**我们强烈推荐您在虚拟环境中安装 PaddleOCR-VL，以避免发生依赖冲突。** 例如，使用 Python venv 标准库创建虚拟环境：

```shell
# 创建虚拟环境
python -m venv .venv_paddleocr
# 激活环境
source .venv_paddleocr/bin/activate
```

执行如下命令完成安装：

```shell
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -U "paddleocr[doc-parser]"
```

> **请注意安装 3.2.0 及以上版本的飞桨框架。**

## 2. 快速开始

当前硬件暂不支持“本地直接推理”路径。如需使用当前硬件加速方案，请继续阅读下一节，采用 vLLM 推理服务路径。

## 3. 使用 VLM 推理服务

本节介绍如何通过 VLM 推理服务完成“客户端 + VLM 推理服务”的组合路径。在当前硬件文档中，示例使用 vLLM 作为 VLM 推理服务后端。

### 3.1 启动 VLM 推理服务

> IMPORTANT:
> 按照本节说明启动的服务仅负责 PaddleOCR-VL 流程中的 VLM 推理环节，不提供完整的端到端文档解析 API。强烈不建议直接通过 HTTP 请求或使用 OpenAI 客户端调用该服务处理文档图像。若您需要部署具备 PaddleOCR-VL 完整能力的服务，请参考后文的服务化部署部分。

**当前硬件支持的启动方式**

| 启动方式 | 状态 | 说明 |
| --- | --- | --- |
| 官方 Docker 镜像 | 支持并提供步骤 | 本节提供 vLLM 推理服务的启动步骤。 |
| 通过 PaddleOCR CLI 安装依赖后启动 | 当前不支持 | 当前硬件不支持该路径。 |
| 直接使用推理加速框架启动 | 未验证 | 当前硬件可通过 vLLM 后端启动 VLM 推理服务，但尚未验证直接使用 vLLM 原生方式启动的路径。 |

PaddleOCR 提供了 Docker 镜像，用于快速启动 vLLM 推理服务。可使用以下命令启动服务（要求 Docker 版本 >= 19.03）：

```shell
docker run -it \
  --name paddleocr_vllm \
  --user root \
  --device /dev:/dev \
  --shm-size 64g \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-intel-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-intel-gpu`（镜像的大小约为 30 GB）更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-intel-gpu-offline`（镜像的大小约为 32 GB）。

启动 vLLM 推理服务时，我们提供了一套默认参数设置。如果您有调整显存占用等更多参数的需求，可以自行配置更多参数。请参考 [3.3.1 服务端参数调整](./PaddleOCR-VL.md#331) 创建配置文件，然后将该文件挂载到容器中，并在启动服务的命令中使用 `backend_config` 指定配置文件，例如：

```shell
docker run -it \
  --name paddleocr_vllm \
  --user root \
  --device /dev:/dev \
  --shm-size 64g \
  --network host \
  -v ./vllm_config.yml:/tmp/vllm_config.yml \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-intel-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/vllm_config.yml
```

> TIP:
> 标签后缀为 `latest-xxx` 的镜像对应最新版本。
> 如果本地已经存在对应的 `latest` 镜像，但希望使用最新功能或修复，建议在继续使用前重新执行一次 `docker pull` 更新镜像。
> 如果希望使用特定版本 PaddleOCR 对应的镜像，可以将标签中的 `latest` 替换为对应版本号：`paddleocr<major>.<minor>`。
> 例如：
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:paddleocr3.4-intel-gpu-offline`

### 3.2 客户端使用方法

客户端调用方式请参考 [PaddleOCR-VL 使用教程 - 3.2 客户端使用方法](./PaddleOCR-VL.md#32)。

### 3.3 性能调优

请参考[PaddleOCR-VL 使用教程 - 3.3 性能调优](./PaddleOCR-VL.md#33)。

## 4. 服务化部署

**当前硬件支持的部署方式**

| 部署方式 | 状态 | 说明 |
| --- | --- | --- |
| Docker Compose 部署 | 支持并提供步骤 | 请继续阅读本节的 4.1。 |
| 手动部署 | 当前不支持 | 当前硬件不支持该路径。 |

> IMPORTANT:
> 本节所介绍的 PaddleOCR-VL 服务与上一节中的 VLM 推理服务有所区别：后者仅负责完整流程中的一个环节（即 VLM 推理），并作为前者的底层服务被调用。

### 4.1 使用 Docker Compose 部署

此步骤主要介绍如何使用 Docker Compose 将 PaddleOCR-VL 部署为服务并调用，具体流程如下：

1. 分别从 [此处](https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/deploy/paddleocr_vl_docker/accelerators/intel-gpu/compose.yaml) 和 [此处](https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/deploy/paddleocr_vl_docker/accelerators/intel-gpu/.env) 获取 Compose 文件与环境变量配置文件并下载到本地。

2. 在 `compose.yaml` 和 `.env` 文件所在目录下执行以下命令启动服务器，默认监听 **8080** 端口：

    ```shell
    # 必须在 compose.yaml 和 .env 文件所在的目录中执行
    docker compose up
    ```

    > 提示：
    > `compose.yaml` 中使用的镜像标签通常由 `.env` 中的 `API_IMAGE_TAG_SUFFIX` 和 `VLM_IMAGE_TAG_SUFFIX` 控制，默认使用 `latest-intel-gpu-offline` 等标签。如需确保拉取到最新的 `latest` 镜像，可先在当前目录执行 `docker compose pull`，再执行 `docker compose up`。
    > 如果希望使用特定版本 PaddleOCR 对应的镜像，可将这两个环境变量中的 `latest` 替换为对应版本号 `paddleocr<major>.<minor>`，例如 `paddleocr3.3-intel-gpu-offline`。

    启动后将看到类似如下输出：

    ```text
    paddleocr-vl-api             | INFO:     Started server process [1]
    paddleocr-vl-api             | INFO:     Waiting for application startup.
    paddleocr-vl-api             | INFO:     Application startup complete.
    paddleocr-vl-api             | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
    ```

此方式基于 vLLM 框架对 VLM 推理进行加速，更适合生产环境部署。

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
<summary>2. 指定 PaddleOCR-VL 服务所使用的 GPU</summary>

编辑 <code>compose.yaml</code> 文件中的 <code>environment</code> 来更改所使用的 GPU。例如，如果您需要使用卡 1 进行部署，可以进行以下修改：

```diff
  paddleocr-vl-api:
    ...
    environment:
+     - XPU_AFFINITY_MASK: 1
    ...
  paddleocr-vlm-server:
    ...
    environment:
+     - XPU_AFFINITY_MASK: 1
    ...
```


</details>

<details>
<summary>3. 调整 VLM 服务端配置</summary>

若您想调整 VLM 服务端的配置，可以参考 <a href="./PaddleOCR-VL.md#331">3.3.1 服务端参数调整</a> 生成配置文件。

生成配置文件后，将以下的 <code>paddleocr-vlm-server.volumes</code> 和 <code>paddleocr-vlm-server.command</code> 字段增加到您的 <code>compose.yaml</code> 中。请将 <code>/path/to/your_config.yaml</code> 替换为您的实际配置文件路径。

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. 调整产线相关配置（如模型路径、批处理大小、部署设备等）</summary>

参考 <a href="./PaddleOCR-VL.md#44">4.4 产线配置调整说明</a> 小节。

</details>

### 4.2 客户端调用方式

请参考[PaddleOCR-VL 使用教程 - 4.3 客户端调用方式](./PaddleOCR-VL.md#43)。

### 4.3 产线配置调整说明

请参考[PaddleOCR-VL 使用教程 - 4.4 产线配置调整说明](./PaddleOCR-VL.md#44)。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程 - 5. 模型微调](./PaddleOCR-VL.md#5)。
